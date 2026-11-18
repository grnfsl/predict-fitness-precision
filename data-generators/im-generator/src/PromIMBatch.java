import java.io.*;
import java.util.*;

import org.deckfour.xes.classification.XEventClassifier;
import org.deckfour.xes.info.impl.XLogInfoImpl;
import org.deckfour.xes.model.XLog;

import org.processmining.acceptingpetrinet.models.AcceptingPetriNet;
import org.processmining.acceptingpetrinet.models.impl.AcceptingPetriNetFactory;

import org.processmining.framework.packages.PackageManager.Canceller;

import org.processmining.models.connections.GraphLayoutConnection;
import org.processmining.models.graphbased.directed.petrinet.Petrinet;
import org.processmining.models.graphbased.directed.petrinet.PetrinetGraph;
import org.processmining.models.semantics.petrinet.Marking;

import org.processmining.plugins.pnml.base.Pnml;
import org.processmining.plugins.pnml.base.FullPnmlElementFactory;

/*
 * Legacy Inductive Miner
 */
import org.processmining.plugins.InductiveMiner.mining.MiningParametersIMf;
import org.processmining.plugins.InductiveMiner.plugins.IMPetriNet;
import org.processmining.plugins.InductiveMiner.plugins.IMProcessTree;
import org.processmining.plugins.InductiveMiner.reduceacceptingpetrinet.ReduceAcceptingPetriNetKeepLanguage;

/*
 * Newer inductiveminer2 implementation
 */
import org.processmining.plugins.InductiveMiner.efficienttree.EfficientTree;
import org.processmining.plugins.InductiveMiner.efficienttree.EfficientTreeReduce;
import org.processmining.plugins.InductiveMiner.efficienttree.EfficientTreeReduceParametersForPetriNet;
import org.processmining.plugins.InductiveMiner.efficienttree.EfficientTree2AcceptingPetriNet;

import org.processmining.plugins.inductiveminer2.logs.IMLog;
import org.processmining.plugins.inductiveminer2.mining.InductiveMiner;
import org.processmining.plugins.inductiveminer2.plugins.InductiveMinerPlugin;
import org.processmining.plugins.inductiveminer2.variants.MiningParametersIMInfrequent;

/*
 * Legacy process-tree -> Petri-net conversion
 */
import org.processmining.processtree.ProcessTree;
import org.processmining.processtree.conversion.ProcessTree2Petrinet;
import org.processmining.processtree.conversion.ProcessTree2Petrinet.PetrinetWithMarkings;


public class PromIMBatch {

    public static void main(String[] args) throws Exception {

        /*
         * Large BPIC XES files can exceed the JAXP defaults of newer JDKs.
         * 0 means unlimited.
         *
         * Only do this for trusted XML/XES input.
         */
        System.setProperty(
                "jdk.xml.maxGeneralEntitySizeLimit",
                "0"
        );

        System.setProperty(
                "jdk.xml.totalEntitySizeLimit",
                "0"
        );


        /*
         * Existing arguments:
         *
         *   1. input XES directory
         *   2. output directory
         *   3. comma-separated noise thresholds
         *
         * Optional new argument:
         *
         *   4. discovery implementation
         *
         *      legacy-plugin
         *      legacy-manual
         *      new-plugin
         *      new-manual
         *
         * If omitted, new-manual is used so that the behaviour remains
         * closest to the original PromIMBatch implementation.
         */
        if (args.length != 3 && args.length != 4) {

            System.err.println(
                    "Usage: java PromIMBatch "
                    + "<input_xes_dir> "
                    + "<output_dir> "
                    + "<thresholds_csv> "
                    + "[legacy-plugin|legacy-manual|new-plugin|new-manual]"
            );

            System.exit(1);
        }


        File inputDir = new File(args[0]);
        File outputDir = new File(args[1]);

        String[] thresholds =
                args[2].split(",");


        /*
         * Preserve the behaviour of the old program if the fourth argument
         * is not given.
         */
        String discoveryMethod =
                args.length == 4
                        ? args[3].trim().toLowerCase(Locale.ROOT)
                        : "new-manual";


        Set<String> supportedMethods =
                new LinkedHashSet<>(Arrays.asList(
                        "legacy-plugin",
                        "legacy-manual",
                        "new-plugin",
                        "new-manual"
                ));


        if (!supportedMethods.contains(discoveryMethod)) {

            throw new IllegalArgumentException(
                    "Unknown discovery method: "
                    + discoveryMethod
                    + ". Supported methods: "
                    + supportedMethods
            );
        }


        if (!inputDir.isDirectory()) {

            throw new IllegalArgumentException(
                    "Input directory does not exist: "
                    + inputDir.getAbsolutePath()
            );
        }


        if (!outputDir.exists() && !outputDir.mkdirs()) {

            throw new RuntimeException(
                    "Could not create output directory: "
                    + outputDir.getAbsolutePath()
            );
        }


        System.out.println(
                "Discovery method: "
                + discoveryMethod
        );


        /*
         * Status CSV
         */
        File statusFile =
                new File(
                        outputDir,
                        "discovery_status.csv"
                );


        PrintWriter status =
                new PrintWriter(
                        new FileWriter(statusFile)
                );


        status.println(
                "log_id,"
                + "log_path,"
                + "threshold,"
                + "discovery_method,"
                + "status,"
                + "output_pnml,"
                + "process_tree,"
                + "runtime_ms,"
                + "error"
        );


        /*
         * Find XES files
         */
        File[] logs =
                inputDir.listFiles(
                        (dir, name) ->
                                name.toLowerCase(Locale.ROOT)
                                        .endsWith(".xes")
                );


        if (logs == null || logs.length == 0) {

            status.close();

            throw new RuntimeException(
                    "No .xes files found in "
                    + inputDir.getAbsolutePath()
            );
        }


        Arrays.sort(
                logs,
                Comparator.comparing(File::getName)
        );


        PromIMBatch miner =
                new PromIMBatch();


        /*
         * Process each log
         */
        for (File logFile : logs) {

            String logId =
                    logFile.getName()
                            .replaceFirst(
                                    "(?i)\\.xes$",
                                    ""
                            );


            File modelDir =
                    new File(
                            outputDir,
                            logId
                    );


            if (!modelDir.exists()
                    && !modelDir.mkdirs()) {

                throw new RuntimeException(
                        "Could not create model directory: "
                        + modelDir.getAbsolutePath()
                );
            }


            /*
             * Process every requested noise threshold
             */
            for (String thresholdText : thresholds) {

                double threshold =
                        Double.parseDouble(
                                thresholdText.trim()
                        );


                String thrName =
                        String.format(
                                Locale.US,
                                "thr_%02d",
                                (int) Math.round(
                                        threshold * 10.0
                                )
                        );


                /*
                 * Include discovery method in the output name so that
                 * models created by different IM implementations do not
                 * overwrite one another.
                 */
                File outPnml =
                        new File(
                                modelDir,
                                logId
                                + "_"
                                + thrName
                                + "_"
                                + discoveryMethod
                                + ".pnml"
                        );


                System.out.println();
                System.out.println(
                        "============================================================"
                );

                System.out.println(
                        "Log: "
                        + logFile.getAbsolutePath()
                );

                System.out.println(
                        "Threshold: "
                        + threshold
                );

                System.out.println(
                        "Discovery method: "
                        + discoveryMethod
                );


                long start =
                        System.currentTimeMillis();


                try {

                    String processTreeString =
                            miner.discoverAndExport(
                                    logFile.getAbsolutePath(),
                                    threshold,
                                    outPnml.getAbsolutePath(),
                                    discoveryMethod
                            );


                    long runtime =
                            System.currentTimeMillis()
                            - start;


                    status.printf(
                            Locale.US,
                            "%s,%s,%.2f,%s,%s,%s,%s,%d,%s%n",

                            csv(logId),

                            csv(
                                    logFile.getAbsolutePath()
                            ),

                            threshold,

                            csv(discoveryMethod),

                            csv("success"),

                            csv(
                                    outPnml.getAbsolutePath()
                            ),

                            csv(processTreeString),

                            runtime,

                            csv("")
                    );


                    System.out.println(
                            "PNML: "
                            + outPnml.getAbsolutePath()
                    );

                    System.out.println(
                            "Runtime: "
                            + runtime
                            + " ms"
                    );


                } catch (Exception e) {

                    long runtime =
                            System.currentTimeMillis()
                            - start;


                    status.printf(
                            Locale.US,
                            "%s,%s,%.2f,%s,%s,%s,%s,%d,%s%n",

                            csv(logId),

                            csv(
                                    logFile.getAbsolutePath()
                            ),

                            threshold,

                            csv(discoveryMethod),

                            csv("failed"),

                            csv(
                                    outPnml.getAbsolutePath()
                            ),

                            csv(""),

                            runtime,

                            csv(
                                    cleanError(e)
                            )
                    );


                    System.err.println(
                            "FAILED: "
                            + cleanError(e)
                    );


                    e.printStackTrace(
                            System.err
                    );
                }


                status.flush();
            }
        }


        status.close();


        System.out.println();
        System.out.println(
                "Finished."
        );

        System.out.println(
                "Status file: "
                + statusFile.getAbsolutePath()
        );
    }


    /*
     * ----------------------------------------------------------------------
     * Main discovery entry point
     * ----------------------------------------------------------------------
     */

    public String discoverAndExport(
            String inputLogPath,
            double noiseThreshold,
            String outputPnmlPath,
            String discoveryMethod)
            throws Exception {


        /*
         * Read the log exactly once.
         */
        XLog log =
                readSingleLog(
                        inputLogPath
                );


        /*
         * Use concept:name, matching the normal IM configuration.
         */
        XEventClassifier classifier =
                XLogInfoImpl.NAME_CLASSIFIER;


        /*
         * Select the requested IM implementation.
         */
        DiscoveryResult result =
                discover(
                        log,
                        noiseThreshold,
                        classifier,
                        discoveryMethod
                );


        AcceptingPetriNet petri =
                result.acceptingPetriNet;


        if (petri == null) {

            throw new RuntimeException(
                    "Inductive Miner returned "
                    + "null AcceptingPetriNet."
            );
        }


        if (petri.getNet() == null) {

            throw new RuntimeException(
                    "Inductive Miner returned "
                    + "an AcceptingPetriNet "
                    + "with null Petri net."
            );
        }


        Petrinet net =
                petri.getNet();


        Marking initialMarking =
                petri.getInitialMarking();


        Collection<Marking> finalMarkings =
                petri.getFinalMarkings();


        /*
         * Use your existing PNML exporter.
         */
        exportPetrinetToPNMLFile(
                net,
                initialMarking,
                finalMarkings,
                outputPnmlPath
        );


        System.out.println(
                "Process Model Discovered using "
                + discoveryMethod
        );


        return result.processTreeString;
    }


    /*
     * ----------------------------------------------------------------------
     * Discovery method dispatcher
     * ----------------------------------------------------------------------
     */

    private DiscoveryResult discover(
            XLog log,
            double noiseThreshold,
            XEventClassifier classifier,
            String method)
            throws Exception {


        switch (method) {

            case "legacy-plugin":

                return mineLegacyPlugin(
                        log,
                        noiseThreshold,
                        classifier
                );


            case "legacy-manual":

                return mineLegacyManual(
                        log,
                        noiseThreshold,
                        classifier
                );


            case "new-plugin":

                return mineNewPlugin(
                        log,
                        noiseThreshold,
                        classifier
                );


            case "new-manual":

                return mineNewManual(
                        log,
                        noiseThreshold,
                        classifier
                );


            default:

                throw new IllegalArgumentException(
                        "Unsupported discovery method: "
                        + method
                );
        }
    }


    /*
     * ----------------------------------------------------------------------
     * Legacy ProM IMf - plugin implementation
     * ----------------------------------------------------------------------
     *
     * This calls IMPetriNet.minePetriNet(), i.e. the legacy helper used by
     * ProM's Inductive Miner Petri-net plugin.
     */

    private DiscoveryResult mineLegacyPlugin(
            XLog log,
            double noiseThreshold,
            XEventClassifier classifier)
            throws Exception {


        MiningParametersIMf parameters =
                createLegacyParameters(
                        noiseThreshold,
                        classifier
                );


        Object[] result =
                IMPetriNet.minePetriNet(
                        log,
                        parameters,
                        neverCancelled()
                );


        if (result == null
                || result.length < 3) {

            throw new RuntimeException(
                    "Legacy IM plugin returned "
                    + "an incomplete result."
            );
        }


        if (!(result[0] instanceof Petrinet)
                || !(result[1] instanceof Marking)
                || !(result[2] instanceof Marking)) {

            throw new RuntimeException(
                    "Legacy IM plugin returned "
                    + "unexpected result types."
            );
        }


        Petrinet net =
                (Petrinet) result[0];


        Marking initialMarking =
                (Marking) result[1];


        Marking finalMarking =
                (Marking) result[2];


        AcceptingPetriNet acceptingNet =
                AcceptingPetriNetFactory
                        .createAcceptingPetriNet(
                                net,
                                initialMarking,
                                finalMarking
                        );


        /*
         * IMPetriNet returns the Petri net and markings, but it does not
         * return the intermediate process tree.
         *
         * Therefore process_tree is left empty instead of mining twice.
         */
        return new DiscoveryResult(
                acceptingNet,
                ""
        );
    }


    /*
     * ----------------------------------------------------------------------
     * Legacy ProM IMf - manual implementation
     * ----------------------------------------------------------------------
     *
     * Process:
     *
     *   XLog
     *      |
     *      v
     * IMProcessTree
     *      |
     *      v
     * ProcessTree
     *      |
     *      v
     * ProcessTree2Petrinet
     *      |
     *      v
     * AcceptingPetriNet
     *      |
     *      v
     * language-preserving reduction
     */

    private DiscoveryResult mineLegacyManual(
            XLog log,
            double noiseThreshold,
            XEventClassifier classifier)
            throws Exception {


        MiningParametersIMf parameters =
                createLegacyParameters(
                        noiseThreshold,
                        classifier
                );


        Canceller canceller =
                neverCancelled();


        ProcessTree tree =
                IMProcessTree.mineProcessTree(
                        log,
                        parameters,
                        canceller
                );


        if (tree == null) {

            throw new RuntimeException(
                    "Legacy Inductive Miner "
                    + "returned null process tree."
            );
        }


        PetrinetWithMarkings converted =
                ProcessTree2Petrinet.convert(
                        tree
                );


        if (converted == null
                || converted.petrinet == null
                || converted.initialMarking == null
                || converted.finalMarking == null) {

            throw new RuntimeException(
                    "Could not convert legacy "
                    + "process tree to Petri net."
            );
        }


        AcceptingPetriNet acceptingNet =
                AcceptingPetriNetFactory
                        .createAcceptingPetriNet(
                                converted.petrinet,
                                converted.initialMarking,
                                converted.finalMarking
                        );


        /*
         * Same language-preserving reduction used by the attached
         * implementation.
         */
        ReduceAcceptingPetriNetKeepLanguage.reduce(
                acceptingNet,
                canceller
        );


        return new DiscoveryResult(
                acceptingNet,
                tree.toString()
        );
    }


    /*
     * ----------------------------------------------------------------------
     * New inductiveminer2 IMf - plugin implementation
     * ----------------------------------------------------------------------
     */

    private DiscoveryResult mineNewPlugin(
            XLog log,
            double noiseThreshold,
            XEventClassifier classifier)
            throws Exception {


        MiningParametersIMInfrequent parameters =
                createNewParameters(
                        noiseThreshold,
                        classifier
                );


        IMLog imlog =
                parameters.getIMLog(
                        log
                );


        AcceptingPetriNet acceptingNet =
                InductiveMinerPlugin.minePetriNet(
                        imlog,
                        parameters,
                        neverCancelled()
                );


        if (acceptingNet == null
                || acceptingNet.getNet() == null) {

            throw new RuntimeException(
                    "New IM plugin returned "
                    + "null AcceptingPetriNet."
            );
        }


        /*
         * Plugin returns the final net directly.
         * The process tree is not exposed.
         */
        return new DiscoveryResult(
                acceptingNet,
                ""
        );
    }


    /*
     * ----------------------------------------------------------------------
     * New inductiveminer2 IMf - manual implementation
     * ----------------------------------------------------------------------
     *
     * This is the implementation closest to your original PromIMBatch.
     */

    private DiscoveryResult mineNewManual(
            XLog log,
            double noiseThreshold,
            XEventClassifier classifier)
            throws Exception {


        MiningParametersIMInfrequent parameters =
                createNewParameters(
                        noiseThreshold,
                        classifier
                );


        IMLog imlog =
                parameters.getIMLog(
                        log
                );


        Canceller canceller =
                neverCancelled();


        /*
         * Mine the EfficientTree.
         */
        EfficientTree tree =
                InductiveMiner.mineEfficientTree(
                        imlog,
                        parameters,
                        canceller
                );


        if (tree == null) {

            throw new RuntimeException(
                    "New Inductive Miner returned "
                    + "null EfficientTree."
            );
        }


        /*
         * Same reduction you already had.
         */
        EfficientTreeReduce.reduce(
                tree,
                new EfficientTreeReduceParametersForPetriNet(
                        false
                )
        );


        String processTreeString =
                tree.toString();


        /*
         * Convert to accepting Petri net.
         */
        AcceptingPetriNet acceptingNet =
                EfficientTree2AcceptingPetriNet.convert(
                        tree
                );


        if (acceptingNet == null
                || acceptingNet.getNet() == null) {

            throw new RuntimeException(
                    "Could not convert EfficientTree "
                    + "to AcceptingPetriNet."
            );
        }


        /*
         * The attached implementation additionally applies the same
         * language-preserving Petri-net reduction after conversion.
         */
        ReduceAcceptingPetriNetKeepLanguage.reduce(
                acceptingNet,
                canceller
        );


        return new DiscoveryResult(
                acceptingNet,
                processTreeString
        );
    }


    /*
     * ----------------------------------------------------------------------
     * Legacy IM parameters
     * ----------------------------------------------------------------------
     */

    private MiningParametersIMf createLegacyParameters(
            double noiseThreshold,
            XEventClassifier classifier) {


        MiningParametersIMf parameters =
                new MiningParametersIMf();


        parameters.setNoiseThreshold(
                (float) noiseThreshold
        );


        parameters.setClassifier(
                classifier
        );


        return parameters;
    }


    /*
     * ----------------------------------------------------------------------
     * New IM parameters
     * ----------------------------------------------------------------------
     */

    private MiningParametersIMInfrequent createNewParameters(
            double noiseThreshold,
            XEventClassifier classifier) {


        MiningParametersIMInfrequent parameters =
                new MiningParametersIMInfrequent();


        parameters.setNoiseThreshold(
                (float) noiseThreshold
        );


        parameters.setClassifier(
                classifier
        );


        return parameters;
    }


    /*
     * ----------------------------------------------------------------------
     * Canceller
     * ----------------------------------------------------------------------
     */

    private static Canceller neverCancelled() {

        return new Canceller() {

            @Override
            public boolean isCancelled() {

                return false;
            }
        };
    }


    /*
     * ----------------------------------------------------------------------
     * XES reader
     * ----------------------------------------------------------------------
     */

    public XLog readSingleLog(
            String fileName)
            throws Exception {


        File initialFile =
                new File(
                        fileName
                );


        org.xeslite.parser.XesLiteXmlParser parser =
                new org.xeslite.parser.XesLiteXmlParser(
                        true
                );


        try (InputStream inputStream =
                     new FileInputStream(initialFile)) {


            List<XLog> parsedLogs =
                    parser.parse(
                            inputStream
                    );


            if (parsedLogs != null
                    && !parsedLogs.isEmpty()) {

                return parsedLogs.get(0);
            }
        }


        throw new RuntimeException(
                "Could not read log: "
                + fileName
        );
    }


    /*
     * ----------------------------------------------------------------------
     * PNML exporter
     * ----------------------------------------------------------------------
     *
     * This is essentially your existing PNML export code.
     */

    public void exportPetrinetToPNMLFile(
            PetrinetGraph net,
            Marking marking,
            Collection<Marking> finalMarkings,
            String targetName)
            throws Exception {


        GraphLayoutConnection layout =
                new GraphLayoutConnection(
                        net
                );


        HashMap<PetrinetGraph, Marking> markedNets =
                new HashMap<
                        PetrinetGraph,
                        Marking
                        >();


        HashMap<
                PetrinetGraph,
                Collection<Marking>
                > finalMarkedNets =
                new HashMap<
                        PetrinetGraph,
                        Collection<Marking>
                        >();


        markedNets.put(
                net,
                marking
        );


        finalMarkedNets.put(
                net,
                finalMarkings
        );


        Pnml pnml =
                new Pnml();


        FullPnmlElementFactory factory =
                new FullPnmlElementFactory();


        synchronized (factory) {

            pnml.setFactory(
                    factory
            );


            pnml =
                    pnml.convertFromNet(
                            markedNets,
                            finalMarkedNets,
                            layout
                    );
        }


        try (PrintWriter out =
                     new PrintWriter(targetName)) {


            out.write(
                    "<?xml version=\"1.0\" "
                    + "encoding=\"ISO-8859-1\"?>\n"
            );


            out.write(
                    pnml.exportElement(
                            pnml
                    )
            );


            out.flush();
        }
    }


    /*
     * ----------------------------------------------------------------------
     * Discovery result
     * ----------------------------------------------------------------------
     */

    private static class DiscoveryResult {

        final AcceptingPetriNet acceptingPetriNet;

        final String processTreeString;


        DiscoveryResult(
                AcceptingPetriNet acceptingPetriNet,
                String processTreeString) {


            this.acceptingPetriNet =
                    acceptingPetriNet;


            this.processTreeString =
                    processTreeString == null
                            ? ""
                            : processTreeString;
        }
    }


    /*
     * ----------------------------------------------------------------------
     * CSV helpers
     * ----------------------------------------------------------------------
     */

    private static String csv(
            String value) {


        if (value == null) {

            return "";
        }


        String escaped =
                value.replace(
                        "\"",
                        "\"\""
                );


        return "\""
                + escaped
                + "\"";
    }


    private static String cleanError(
            Exception e) {


        String msg =
                e.toString();


        return msg
                .replace(
                        "\n",
                        " "
                )
                .replace(
                        "\r",
                        " "
                );
    }
}