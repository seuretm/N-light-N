/*****************************************************
 N-light-N

 A Highly-Adaptable Java Library for Document Analysis with
 Convolutional Auto-Encoders and Related Architectures.

 -------------------
 Author:
 2016 by Mathias Seuret <mathias.seuret@unifr.ch>
 and Michele Alberti <michele.alberti@unifr.ch>
 -------------------

 This software is free software; you can redistribute it and/or
 modify it under the terms of the GNU Lesser General Public
 License as published by the Free Software Foundation version 3.

 This software is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 Lesser General Public License for more details.

 You should have received a copy of the GNU Lesser General Public
 License along with this software; if not, write to the Free Software
 Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 ******************************************************************************/

package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.GroundTruthDataBlock;
import diuf.diva.dia.ms.util.Random;
import diuf.diva.dia.ms.util.Tracer;
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class trains a classifier previously loaded into the script. It it possible to specify different parameters
 * for training. Every classifier must have his own method for being trained as for some the data format might be
 * different.
 *
 * @author Michele Alberti
 */
public class TrainClassifier extends AbstractCommand {

    /**
     * Size of the minibatch training
     */
    private final int BATCHSIZE = 1;
    /**
     * Max number of samples to be evaluated for the training
     */
    private int SAMPLES;
    /**
     * Max amount of minutes available for the training
     */
    private int MAXTIME;
    /**
     * Tracer object, responsible for the creation and support for plots
     */
    private Tracer tracer;
    /**
     * Number of layers to be trained from the top
     */
    private int nbLayers;
    /**
     * Indicates whether the classifier is under continuous evaluation
     */
    private boolean EVALUATE = false;
    /**
     * Denotes how often (after how many samples) an evaluation is run
     */
    private int EVALUATE_STEP;
    /**
     * Denotes how many samples are going to be used for the evaluation
     */
    private int EVALUATE_SIZE;
    /**
     * Writer to write on file the continuous evaluation
     */
    private BufferedWriter bufferedWriter;

    /**
     * Constructor of the class.
     *
     * @param script which creates the command
     */
    public TrainClassifier(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {

        // Fetching classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        nbLayers = classifier.getNumLayers();

        // Fetching an optional parameter who specifies how many parameters should be trained from the top
        if (element.getChild("numLayers") != null) {
            nbLayers = Integer.parseInt(readElement(element, "numLayers"));
        }

        // Parsing parameters of training
        this.SAMPLES = Integer.parseInt(readElement(element, "samples"));
        this.MAXTIME = (element.getChild("max-time") != null) ? Integer.parseInt(readElement(element, "max-time")) : 0;
        if (MAXTIME == 0) {
            MAXTIME = Integer.MAX_VALUE;
        }

        // Parse the continuous evaluation parameters
        if (element.getChild("evaluate") != null) {
            EVALUATE = true;
            Element e = element.getChild("evaluate");
            EVALUATE_STEP = Integer.parseInt(readElement(e, "step"));
            EVALUATE_SIZE = Integer.parseInt(readElement(e, "size"));

            String evaluateOutpath = readElement(e, "output-folder");

            // Saving on disk the output image
            int i = 0;
            File file;
            // Looks for next free number on the output path folder
            while ((file = new File(evaluateOutpath + i + "-evaluate-classifier.txt")).exists()) i++;

            try {
                // Create the path
                file.getParentFile().mkdirs();
                // Create the file
                file.createNewFile();
                bufferedWriter = new BufferedWriter(new FileWriter(file, true));
            } catch (IOException exeption) {
                exeption.printStackTrace();
            }
        }

        // Fetching datasets
        String dataset = readElement(element, "dataset");
        HashMap<Integer, ArrayList<DataBlock>> fnbds = script.fileNameBasedDatasets.get(dataset);
        Dataset ds = script.datasets.get(dataset);

        if (ds == null) {
            XMLScript.print("Dataset:" + dataset + " not found!");
        }

        // Testing if tracer should be init
        tracer = null;
        if (element.getChild("save-progress") != null) {
            try {
                /* Expected points on the plots are the number of expected epochs.
                 * So number of total samples divided by size of epoch times the number of classes per image
                 */
                int expectedSamples = SAMPLES;
                if (fnbds == null) expectedSamples /= (ds.size() * classifier.getOutputSize());
                // Tracer init
//                tracer = new Tracer(
//                        classifier.name() + " training error",
//                        "Epoch",
//                        "Error",
//                        expectedSamples,
//                        element.getChild("display-progress") != null
//                );
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Train the classifier
        XMLScript.println("\"Classifier Starting training[" + ref + "] {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "}");

        long startTime = System.currentTimeMillis();

        switch (classifier.type()) {
            case "pixel":
                if (fnbds == null) {
                    trainPixelBasedClassifiers(classifier, ds);
                } else {
                    fileNameBased(classifier, fnbds);
                }
                break;

            default:
                error("invalid classifier type :" + classifier.type());
        }

        XMLScript.println("Training time = " + (int) (System.currentTimeMillis() - startTime) / 1000.0);
        XMLScript.println("Finish training classifier [" + ref + "]");

        // If tracer has been used
//        if (tracer != null) {
//            // Add the tracks on the tracer
//            tracer.addRawData();
//            tracer.addCumulatedAverage();
//            tracer.addMovingAverage();
//            tracer.addMovingMedian();
//
//            // Show the plot
//            tracer.display();
//
//            // If required, save the plot on disk at the provided path
//            if (element.getChild("save-progress") != null) {
//                try {
//                    tracer.savePlot(readElement(element, "save-progress"));
//                } catch (Exception e) {
//                    e.printStackTrace();
//                }
//            }
//        }

        if (bufferedWriter != null) {
            bufferedWriter.flush();
            bufferedWriter.close();
        }

        return "";
    }

    /**
     * This method is designed to train a classifier with pixel-based dat as input. It could be used also with
     * other data types of course, provided that the interface Classifier.java is properly implemented.
     *
     * XML syntax to use this feature:
     *
     * <train-classifier ref="myClassifier">
     * <dataset>stringID</dataset>   <!-- ID of the test dataset -->
     * <groundTruth>stringID</groundTruth> <!-- ID of the ground truth dataset -->
     * <samples>int</samples>
     * <max-time>int</max-time>
     * <display-progress>200</display-progress> <!-- optional -->
     * <save-progress>stringPATH</save-progress> <!-- optional -->
     *
     * </train-classifier>
     *
     * @param classifier the classifier which is going to be trained
     * @param ds      the dataset containing the images which will be used for training
     *                   is immediately interrupted and the result returned
     */
    private void trainPixelBasedClassifiers(Classifier classifier, Dataset<GroundTruthDataBlock> ds) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Counter that keeps track on how many samples have been already executed
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Counter of epochs (logging purpose only)
        int epoch = 0;

        // Epoch-wise error
        double err;
        int epochSize;

        // Batch handling
        int batch = 0;

        // Verify input size for the whole dataset
        for (int i = 0; i < ds.size(); i++) {
            DataBlock img = ds.get(i);
            if (img.getWidth() < classifier.getInputWidth() || img.getHeight() < classifier.getInputHeight()) {
                throw new Error("an image of the dataset is smaller than the input of the classifier:\n" +
                        "[" + img.getWidth() + "x" + img.getHeight() + "] < ["
                        + classifier.getInputWidth() + "x" + classifier.getInputHeight() + "]"
                );
            }
        }

        // Analyse the images
        for (GroundTruthDataBlock db : ds) {
            db.analyse(classifier.getInputWidth(), classifier.getInputHeight());
            db.subSample((int) Math.ceil(SAMPLES / ds.size()));
        }

        XMLScript.print("Progress[");

        // Train the classifier until enough samples have been evaluated
        classifier.startTraining();
        while (sample < SAMPLES) {

            // Epoch-wise error
            err = 0;
            epochSize = 0;

            // Log every ~10% the progress of training
            if ((sample * 10) / SAMPLES >= loggingProgress) {
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
                System.out.print(loggingProgress * 10 + "%");
                loggingProgress = (sample * 10) / SAMPLES + 1;
            }

            // Iterate over all images in the dataset
            for (GroundTruthDataBlock db : ds) {
                // For each image, take a sample out of each class -> data balancing
                for (int c = 0; c <= db.getNumberOfClasses(); c++) {

                    // Get next representative
                    Pixel p = db.getRandomRepresentative(c);

                    // If pixel 'p' is null it means that this specific GT does not contain this class
                    if (p == null) {
                        continue;
                    }

                    // Set input to classifier
                    classifier.centerInput(db, p.x, p.y);

                    // Forward
                    classifier.compute();

                    // Set the expected values to 0 for all outputs neurons and 1 for the correct class
                    for (int j = 0; j < classifier.getOutputSize(); j++) {
                        classifier.setExpected(j, (j == c) ? 1 : 0);
                    }

                    // Learning the classifier
                    err += classifier.backPropagate(nbLayers);

                    // Increase counters
                    sample++;
                    epochSize++;
                    batch++;

                    // If is the end of the mini-batch
                    if (batch >= BATCHSIZE) {
                        batch = 0;
                        // Update weights
                        classifier.learn(nbLayers);
                    }

                    // If is the case, evaluate the classifier
                    if (sample % EVALUATE_STEP == 0) {
                        try {
                            bufferedWriter.write(sample + "\t" + EvaluateClassifier.printResults(EvaluateClassifier.evaluateDataSet(ds, classifier, EVALUATE_SIZE)) + "\n");
                            bufferedWriter.flush();
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }

                    // Stop execution if MAXTIME reached
                    if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                        // Complete the logging progress
                        System.out.println("]");
                        XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                        return;
                    }
                }
            }

            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err / epochSize);
            }

            // Log the number of epochs
            epoch++;

        }
        classifier.stopTraining();

        // Complete the logging progress
        System.out.println(" 100%]");
    }

    /**
     * This method is designed to train a classifier with the GT on his filename.
     * Furthermore, it is assumed that each input image is same size as the classifier
     * input size. Examples of dataset like this are: CIFAR, MNIST.
     */
    public void fileNameBased(Classifier classifier, HashMap<Integer, ArrayList<DataBlock>> fnbds ) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Counter that keeps track on how many samples have been already executed
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Counter of epochs (logging purpose only)
        int epoch = 0;

        // Epoch-wise error
        double err;
        int epochSize;

        // Batch handling
        int batch = 0;

        // Number of classes found in the ds
        final int nbClasses = fnbds.size();

        // Check that the size of the classifier is equal to the size of the image!
        if ((classifier.getInputWidth() != fnbds.get(0).get(0).getWidth()) ||
                (classifier.getInputHeight() != fnbds.get(0).get(0).getHeight())) {
            throw new RuntimeException(
                    "Classifier input size (" +
                            classifier.getInputWidth() + "x" + classifier.getInputHeight() + ")" +
                            " does not match images size (" +
                            fnbds.get(0).get(0).getWidth() + "x" + fnbds.get(0).get(0).getHeight() + "!");
        }

        /***********************************************************************************************
         * TRAIN THE CLASSIFIER
         **********************************************************************************************/

        XMLScript.print("Progress[");

        // Train the classifier until enough samples have been evaluated
        while (sample < SAMPLES) {

            // Epoch-wise error
            err = 0;
            epochSize = 0;

            // Log every ~10% the progress of training
            if ((sample * 10) / SAMPLES >= loggingProgress) {
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
                System.out.print(loggingProgress * 10 + "%");
                loggingProgress = (sample * 10) / SAMPLES + 1;
            }

            // Iterate over all classes (data balancing)
            for (int c = 0; c < nbClasses; c++) {

                // Set input to classifier
                classifier.setInput(fnbds.get(c).get((int) (Random.nextDouble() * fnbds.get(c).size())), 0, 0);

                // Forward
                classifier.compute();

                // Set the expected values to 0 for all outputs neurons and 1 for the correct class
                for (int j = 0; j < classifier.getOutputSize(); j++) {
                    classifier.setExpected(j, (j == c) ? 1 : 0);
                }

                // Learning the classifier
                err += classifier.backPropagate(nbLayers);

                // Increase counters
                sample++;
                epochSize++;
                batch++;

                // If is the end of the mini-batch
                if (batch >= BATCHSIZE) {
                    batch = 0;
                    // Update weights
                    classifier.learn(nbLayers);
                }

                // Stop execution if MAXTIME reached
                if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                    // Complete the logging progress
                    System.out.println("]");
                    XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");

                    // Free memory
                    fnbds = null;
                    System.gc();

                    return;
                }
            }


            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err / epochSize);
            }

            // Log the number of epochs
            epoch++;

        }

        // Complete the logging progress
        System.out.println(" 100%]");
    }

    @Override
    public String tagName() {
        return "train-classifier";
    }

}
