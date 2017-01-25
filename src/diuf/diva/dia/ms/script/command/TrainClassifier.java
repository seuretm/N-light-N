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
import diuf.diva.dia.ms.util.BiDataBlock;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.Tracer;
import diuf.diva.dia.ms.util.misc.ImageAnalysis;
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import java.io.File;
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
     * Constructor of the class.
     * @param script which creates the command
     */
    public TrainClassifier(XMLScript script) {
        super(script);
    }

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
     * Size of the minibatch training
     */
    private final int BATCHSIZE = 1;
    /**
     * Number of layers to be trained from the top
     */
    private int nbLayers;

    @Override
    public String execute(Element element) throws Exception {

        /** If the tag </filenamebased> is present
         *  call the relative method. Otherwise go on
         *  with typical training.
         */
        if (element.getChild("filenamebased") != null) {
            fileNameBased(element);
            return "";
        }

        // Fetching classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        nbLayers = classifier.getNumLayers();

        // Fetching an optional parameter who specifies how many parameters should be trained from the top
        if (element.getChild("numLayers") != null) {
            nbLayers = Integer.parseInt(readElement(element, "numLayers"));
        }

        // Fetching datasets
        String dataset = readElement(element, "dataset");
        String groundTruth = readElement(element, "groundTruth");

        Dataset ds = script.datasets.get(dataset);
        Dataset gt = script.datasets.get(groundTruth);

        if (ds.size() != gt.size()) {
            error("size of dataset(" + ds.size() + ") and ground-truth(" + gt.size() + ") mismatch");
        }

        // Parsing parameters of training
        this.SAMPLES = Integer.parseInt(readElement(element, "samples"));
        this.MAXTIME = Integer.parseInt(readElement(element, "max-time"));
        if (MAXTIME == 0) {
            MAXTIME = Integer.MAX_VALUE;
        }

        // Testing if tracer should be init
        tracer = null;
        if (element.getChild("save-progress") != null) {
            try {
                /* Expected points on the plots are the number of expected epochs.
                 * So number of total samples divided by size of epoch times the number of classes per image
                 */
                int expectedSamples = SAMPLES / (ds.size() * classifier.getOutputSize());
                // Tracer init
                tracer = new Tracer(
                        classifier.name() + " training error",
                        "Epoch",
                        "Error",
                        expectedSamples,
                        element.getChild("display-progress") != null
                );
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


        // Train the classifier
        script.println(
                "\"SCAE Starting training["
                        + ref
                        + "] {maximum time:"
                        + MAXTIME
                        + "m, samples:"
                        + SAMPLES
                        + "}"
        );

        long startTime = System.currentTimeMillis();

        switch (classifier.type()) {
            case "pixel":
                trainPixelBasedClassifiers(classifier, ds, gt);
                break;

            default:
                error("invalid classifier type :" + classifier.type());
        }

        script.println("Training time = " + (int) (System.currentTimeMillis() - startTime) / 1000.0);
        script.println("Finish training classifier [" + ref + "]");

        // If tracer has been used
        if (tracer != null) {
            // Add the tracks on the tracer
            tracer.addRawData();
            tracer.addCumulatedAverage();
            tracer.addMovingAverage();
            tracer.addMovingMedian();

            // Show the plot
            tracer.display();

            // If required, save the plot on disk at the provided path
            if (element.getChild("save-progress") != null) {
                try {
                    tracer.savePlot(readElement(element, "save-progress"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
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
     * <!-- ID of the test dataset -->
     * <dataset>stringID</dataset>
     * <!-- ID of the ground truth dataset -->
     * <groundTruth>stringID</groundTruth>
     * <samples>int</samples>
     * <max-time>int</max-time>
     * <!-- optional -->
     * <display-progress>200</display-progress>
     * <!-- optional -->
     * <save-progress>stringPATH</save-progress>
     * </train-classifier>
     *
     * @param classifier the classifier which is going to be trained
     * @param dsImg      the dataset containing the images which will be used for training
     * @param dsGt       the dataset containing the ground truth for the provided dataset
     *                   is immediately interrupted and the result returned
     */
    private void trainPixelBasedClassifiers(Classifier classifier, Dataset dsImg, Dataset dsGt) {

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
        for (int i = 0; i < dsImg.size(); i++) {
            DataBlock img = dsImg.get(i);
            if (img.getWidth() < classifier.getInputWidth() || img.getHeight() < classifier.getInputHeight()) {
                throw new Error("an image of the dataset is smaller than the input of the classifier:\n" +
                        "[" + img.getWidth() + "x" + img.getHeight() + "] < ["
                        + classifier.getInputWidth() + "x" + classifier.getInputHeight() + "]"
                );
            }
        }

        // ImageAnalysis init, for data balancing
        ImageAnalysis[] imageAnalyses = new ImageAnalysis[dsImg.size()];
        for (int i = 0; i < dsImg.size(); i++) {
            imageAnalyses[i] = new ImageAnalysis(dsGt.get(i), classifier.getInputWidth(), classifier.getInputHeight());
            imageAnalyses[i].subSample((int) Math.ceil(SAMPLES / dsImg.size()));
        }

        script.print("Progress[");

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

            // Iterate over all images in the dataset
            for (int i = 0; i < dsImg.size(); i++) {
                // For each image, take a sample out of each class -> data balancing
                for (int c = 0; c < imageAnalyses[i].nbClasses; c++) {

                    // Get next representative
                    Pixel p = imageAnalyses[i].getNextRepresentative(c);

                    // If pixel 'p' is null it means that this specific GT does not contain this class
                    if (p == null) {
                        continue;
                    }

                    // Set input to classifier
                    classifier.centerInput(dsImg.get(i), p.x, p.y);

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
                        script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
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

        // Complete the logging progress
        System.out.println(" 100%]");
    }

    /**
     * This method is designed to train a classifier with the GT on his filename.
     * Furthermore, it is assumed that each input image is same size as the classifier
     * input size. Examples of dataset like this are: CIFAR, MNIST.
     * I am sorry, I know this is not the most elegant way to implement but I lack
     * the time to do it better.
     * In the future maybe Dataset will contain the filename of the data it reads
     * and will make possible to merge this method with his Dataset based counterpart.
     * <p>
     * XML syntax to use this feature:
     * <p>
     * <train-classifier ref="myClassifier">
     * <dataset>stringPATH</dataset>           // PATH (and not ID!) of the training data set
     * </filenamebased>
     * <subsampledataset>500</subsampledataset>    // Optional: specifies if the dataset should be subsampled
     * <samples>int</samples>
     * <max-time>int</max-time>
     * <display-progress>200</display-progress> 			// optional
     * <save-progress>stringPATH</save-progress> 			// optional: but make no sense if display progress is not there
     * </train-classifier>
     *
     * @param element the node of the XML directly
     */
    public void fileNameBased(Element element) throws IOException {

        /***********************************************************************************************
         * PARSE ELEMENT FROM XML
         **********************************************************************************************/

        // Fetching classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        nbLayers = classifier.getNumLayers();

        // Fetching an optional parameter who specifies how many parameters should be trained from the top
        if (element.getChild("numLayers") != null) {
            nbLayers = Integer.parseInt(readElement(element, "numLayers"));
        }

        // Fetching datasets - this must be a PATH
        File dsdir = new File(readElement(element, "dataset"));
        if (!dsdir.isDirectory()) {
            throw new RuntimeException("As there is the tag <filenamebased/>, <dataset> MUST contain a string with the path of the dataset (and not the reference of a dataset previously loaded)");
        }

        // Parsing parameters of training
        this.SAMPLES = Integer.parseInt(readElement(element, "samples"));
        this.MAXTIME = Integer.parseInt(readElement(element, "max-time"));
        if (MAXTIME == 0) {
            MAXTIME = Integer.MAX_VALUE;
        }

        // Testing if tracer should be init
        tracer = null;
        if (element.getChild("save-progress") != null) {
            try {
                // Expected points on the plots are the number of expected epochs.
                int expectedSamples = SAMPLES;
                // Tracer init
                tracer = new Tracer(classifier.name() + " training error", "Epoch", "Error", expectedSamples, element.getChild("display-progress") != null);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // Check whether dataset should be sub sampled
        int dsSubSample = 1;
        if (element.getChild("subsampledataset") != null) {
            dsSubSample = Integer.parseInt(readElement(element, "subsampledataset"));
        }

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

        /***********************************************************************************************
         * LOAD DATASET WITH DATA BALANCING
         **********************************************************************************************/

        script.println("Loading dataset from path for <filenamebased/>");

        /**
         * This hashMap stores one arrayList per each new class found. In the arrayList
         * are contained Datablock belonging to that class
         */
        HashMap<Integer, ArrayList<DataBlock>> data = new HashMap<>();
        /**
         * Final number of classes found in the picture
         */
        final int nbClasses;

        // Getting file names on that folder
        File[] listOfFiles = dsdir.listFiles();

        // For each file listed
        for (int i = 0; i < listOfFiles.length; i += dsSubSample) {
            // Get correct class from file name
            int correctClass = Character.getNumericValue(listOfFiles[i].getName().charAt(0));
            // Store in the image in the appropriate ArrayList
            if (!data.containsKey(correctClass)) {
                data.put(correctClass, new ArrayList<>());
            }
            data.get(correctClass).add(new BiDataBlock(dsdir + "/" + listOfFiles[i].getName()));

        }

        script.println("Dataset loaded and processed in: " + (System.currentTimeMillis() - startTime) / 1000 + " seconds");

        nbClasses = data.size();

        // Check that the size of the classifier is equal to the size of the image!
        if ((classifier.getInputWidth() != data.get(0).get(0).getWidth()) ||
                (classifier.getInputHeight() != data.get(0).get(0).getHeight())) {
            throw new RuntimeException(
                    "Classifier input size (" +
                            classifier.getInputWidth() + "x" + classifier.getInputHeight() + ")" +
                            " does not match images size (" +
                            data.get(0).get(0).getWidth() + "x" + data.get(0).get(0).getHeight() + "!");
        }

        /***********************************************************************************************
         * TRAIN THE CLASSIFIER
         **********************************************************************************************/

        script.println("\"Classifier starting training[" + ref + "] {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "}");

        script.print("Progress[");

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
                classifier.setInput(data.get(c).get((int) (XMLScript.getRandom().nextDouble() * data.get(c).size())), 0, 0);

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
                    script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");

                    // Free memory
                    data = null;
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

        script.println("Training time = " + (int) (System.currentTimeMillis() - startTime) / 1000.0);
        script.println("Finish training classifier [" + ref + "]");

        // If tracer has been used
        if (tracer != null) {
            // Add the tracks on the tracer
            tracer.addRawData();
            tracer.addCumulatedAverage();
            tracer.addMovingAverage();
            tracer.addMovingMedian();

            // Show the plot
            tracer.display();

            // If required, save the plot on disk at the provided path
            if (element.getChild("save-progress") != null) {
                try {
                    tracer.savePlot(readElement(element, "save-progress"));
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }
        }

        // Free memory
        data = null;
        System.gc();

    }

    @Override
    public String tagName() {
        return "train-classifier";
    }

}
