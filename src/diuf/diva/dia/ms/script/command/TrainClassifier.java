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
import diuf.diva.dia.ms.util.Tracer;
import org.jdom2.Element;

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
                tracer = new Tracer(classifier.name() + " training error", "Epoch", "Error", expectedSamples, element.getChild("display-progress") != null);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }


        // Train the classifier
        script.println("\"SCAE Starting training[" + ref + "] {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "}");

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
     * <dataset>stringID</dataset>           // ID of the testing data set
     * <groundTruth>stringID</groundTruth>   // ID of the ground truth data set
     * <samples>int</samples>
     * <max-time>int</max-time>
     * <display-progress>200</display-progress> 			// optional
     * <save-progress>stringPATH</save-progress> 			// optional: but make no sense if display progress is not there
     * </train-classifier>
     *
     * @param classifier the classifier which is going to be trained
     * @param dsImg      the dataset containing the images which will be used for training
     * @param dsGt       the dataset containing the ground truth for the provided dataset
     *                   is immediately interrupted and the result returned
     */
    private void trainPixelBasedClassifiers(Classifier classifier, Dataset dsImg, Dataset dsGt) {
        long startTime = System.currentTimeMillis();

        // Counter that keeps track on how many samples have been already executed
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Counter of epochs (logging purpose only)
        int epoch = 0;

        // Epoch-wise error
        double err;
        int epochSize = 0;

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
            imageAnalyses[i] = new ImageAnalysis(dsGt.get(i), classifier);
            imageAnalyses[i].subSample(SAMPLES / dsImg.size());
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
                    Pixel p = imageAnalyses[i].getRandomRepresentative(c);

                    // If pixel 'p' is null it means that this specific GT does not contain this class
                    System.out.println("Doc "+i+", class "+c);
                    if (p == null) {
                        System.out.println("  Not found");
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

    @Override
    public String tagName() {
        return "train-classifier";
    }

    /**
     * This class is used for data balancing while training. It analyses a datablock
     * and stores references to different pixels of different classes such that is possible
     * to get them for training.
     */
    private class ImageAnalysis {

        /**
         * This hashMap stores one arrayList per each new class found. In the arrayList
         * are contained pixels belonging to that class
         */
        private final HashMap<Integer, ArrayList<Pixel>> data = new HashMap<>();
        /**
         * Final number of classes found in the picture
         */
        public final int nbClasses;

        /**
         * Creates a image analysis
         *
         * @param gt         the images that need to be analysed
         * @param classifier the classifier which is going to be trained with. Needed for handling
         *                   the borders of the images and also the number of classes to be detected.
         */
        public ImageAnalysis(final DataBlock gt, final Classifier classifier) {
            long startTime = System.currentTimeMillis();

            // Init of the final field
            this.nbClasses = classifier.getOutputSize();

            /* Population of the data keeping in consideration to skip borders of image (because if
             * we then center the input we want the input patch to be within the border of the image!
             */
            int index = gt.getDepth()-1;
            for (int x = classifier.getInputWidth() / 2; x <= gt.getWidth() - classifier.getInputWidth(); x++) {
                for (int y = classifier.getInputHeight() / 2; y <= gt.getHeight() - classifier.getInputHeight(); y++) {
                    int correctClass = Math.round((gt.getValues(x, y)[index] + 1) * 255 / 2.0f);
                    if (!data.containsKey(correctClass)) {
                        data.put(correctClass, new ArrayList<>());
                    }
                    data.get(correctClass).add(new Pixel(x, y));
                }
            }

            // Log creation
            script.println("ImageAnalysis created in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());
            script.println("Number of elements:");
            for (int key : data.keySet()) {
                script.println("  class "+key+": "+data.get(key).size());
            }
        }

        /**
         * Sub-sample (or super sumple if there are too few!) the list to a specific total amount of pixels. This is done to prevent storing in memory
         * millions of pixels unnecessarily
         *
         * @param samples the TOTAL amount of sample that this object will store, equally divided among classes
         */
        public void subSample(int samples) {
            long startTime = System.currentTimeMillis();

            // For each class
            for (int c = 0; c < nbClasses; c++) {
                // Init the new array list for storing the sub sampled points
                ArrayList<Pixel> ssp = new ArrayList<>();

                // Populate the sub sampled points with random representative until we have enough
                for (int i = 0; i < samples / nbClasses; i++) {
                    ssp.add(getRandomRepresentative(c));
                }

                // Remove old array list and set the new one
                data.remove(c);
                data.put(c, ssp);
            }
            System.gc();

            // Log sub sampling
            script.println("ImageAnalysis sub-sampled in: " + (int) (System.currentTimeMillis() - startTime) / 1000.0 + " sec " + this.toString());

        }

        /**
         * Select, return the a random pixel on the list.
         *
         * @param c the class of which the representative will be chosen
         * @return the next pixel on the list
         */
        public Pixel getRandomRepresentative(int c) {
            return (data.get(c) != null) ? data.get(c).get((int) (Math.random() * data.get(c).size())) : null;
        }

        @Override
        public String toString() {
            StringBuilder s = new StringBuilder();
            s.append("[");
            for (int i = 0; i < nbClasses; i++) {
                s.append((data.get(i) != null) ? data.get(i).size() : 0);
                if (i < nbClasses - 1) {
                    s.append(",");
                }

            }
            s.append("]");
            return s.toString();
        }


    }

    /**
     * Support class for easier data structure modelling
     */
    private class Pixel {
        /**
         * X coordinate of the pixel
         */
        public final int x;
        /**
         * Y coordinate of the pixel
         */
        public final int y;

        public Pixel(int x, int y) {
            this.x = x;
            this.y = y;
        }

        @Override
        public String toString() {
            StringBuilder s = new StringBuilder();
            s.append("p[");
            s.append(x);
            s.append(",");
            s.append(y);
            s.append("]");
            return s.toString();
        }
    }
}
