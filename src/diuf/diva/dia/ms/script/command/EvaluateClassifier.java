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
import org.jdom2.Element;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

/**
 * Evaluates the classification accuracy and stores the classification result.
 * Syntax for the XML command:
 *
 *  <evaluate-classifier ref="">
 *      <!-- ID of the dataset to be tested -->
 *      <dataset>stringID</dataset>
 *      <!-- ID of the ground truth dataset -->
 *      <groundTruth>stringID</groundTruth>
 *      <!-- if different from 1, skips some pixels when evaluating -->
 *      <offset-x>int</offset-x>
 *      <offset-y>int</offset-y>
 *      <!-- single or multi-class evaluation -->
 *      <method>enum(single-class,multiple-classes)</method>
 *      <!-- path where result images should be stored -->
 *      <output-folder>stringPATH</output-folder>
 *  </evaluate-classifier>
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class EvaluateClassifier extends AbstractCommand {

    /*  Result expected:
     *  BLACK (0x000000) [0][0] = True negative
     *  BLUE  (0x0000FF) [0][1] = False negative
     *  RED   (0xFF0000) [1][0] = False positive
     *  GREEN (0x00FF00) [1][1] = True positive
     *
     *  WHITE (0xFFFFFF)        = Not evaluated (borders only!)
     *  Optimal results are only black and green
     */
    private final int[][] resultColor = {
            {0x000000, 0x0000FF},
            {0xFF0000, 0x00FF00}
    };

    private enum ErrorType {
        UNDEF,
        SINGLE_CLASS,
        MULTIPLE_CLASSES
    }

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    private boolean printOutputFiles = false;

    public EvaluateClassifier(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {

        /** If the tag </filenamebased> is present
         *  call the relative method. Otherwise go on
         *  with typical training.
         */
        if (element.getChild("filenamebased") != null) {
            return fileNameBased(element);
        }

        // Fetching the classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        if (classifier == null) {
            error("cannot find classifier :" + ref);
        }

        // Fetching the datasets
        Dataset ds = script.datasets.get(readElement(element, "dataset"));
        Dataset gt = script.datasets.get(readElement(element, "groundTruth"));

        if (ds.size() != gt.size()) {
            error("size of dataset(" + ds.size() + ") and ground-truth(" + gt.size() + ") mismatch");
        }

        // Parsing the offsets
        int offsetX = Integer.parseInt(readElement(element, "offset-x"));
        int offsetY = Integer.parseInt(readElement(element, "offset-y"));

        // Parsing the error types
        ErrorType et = ErrorType.UNDEF;

        String method = readElement(element, "method");

        if (method.equals("single-class")) {
            et = ErrorType.SINGLE_CLASS;
        }
        if (method.equals("multiple-classes")) {
            et = ErrorType.MULTIPLE_CLASSES;
        }
        if (et == ErrorType.UNDEF) {
            error("invalid method, use either single-class or multiple-classes tag");
        }

        // Parsing and verifying the output folder
        String outPath = null;
        if (element.getChild("output-folder") != null) {
            printOutputFiles = true;
            outPath = readElement(element, "output-folder");
            File file = new File(outPath);
            if (!file.exists()) {
                file.mkdirs();
            }
            if (!file.isDirectory()) {
                error(outPath + " is not a folder");
            }
        }

        // Starting
        script.println("Start evaluating classifier: " + classifier.name());

        float[] cumulatedError = {0, 0, 0};
        for (int i = 0; i < ds.size(); i++) {

            switch (et) {
                case SINGLE_CLASS:
                    float e = getSingleClassError(ds.get(i), gt.get(i), classifier, offsetX, offsetY, outPath + "/" + classifier.name() + "-" + i);
                    script.println("Classified image " + (i + 1) + "/" + ds.size() + " : ACC=" + String.format("%.3f", e));
                    cumulatedError[0] += e;
                    break;

                case MULTIPLE_CLASSES:
                    script.print("Classified image " + (i + 1) + "/" + ds.size() + " :");
                    float[] rv = getMultipleClassesError(ds.get(i), gt.get(i), classifier, offsetX, offsetY, outPath + "/" + classifier.name() + "-" + i);
                    cumulatedError[0] += rv[0];
                    cumulatedError[1] += rv[1];
                    cumulatedError[2] += rv[2];
                    break;
            }
        }

        // Print results over all images
        switch (et) {
            case SINGLE_CLASS:
                script.println("Finish evaluating classifier single-class: ACC=" + String.format("%.2f", cumulatedError[0] / ds.size()));
                break;
            case MULTIPLE_CLASSES:
                script.println("Finish evaluating classifier multiple-classes: PRE=" + String.format("%.2f", cumulatedError[0] / ds.size()) + " REC=" + String.format("%.2f", cumulatedError[1] / ds.size()) + " F1=" + String.format("%.2f", cumulatedError[2] / ds.size()));
                break;
        }

        return String.valueOf(cumulatedError[0] / ds.size());
    }

    /**
     * This methods evaluates the classifier expecting single classes prediction. The cumulated error is computed as
     * sum of all misclassified evaluation. Every mistake count as 1. Optimally the output result image is fully green
     *
     * @param img        the image we will use as input
     * @param gt         the ground truth for the image provided as input
     * @param classifier the object classifier which has been selected to evaluate
     * @param ox         the X-axis offset which will be used to select the next pixel to be evaluated. In fact this methods
     *                   does not evaluate all the pixels but only one in every patch defined by ox and oy. This way it is
     *                   possible to execute a rough evaluation of the classifier without running every pixel of the imaged
     *                   trough the classification process (which would be slow). If ox and oy are set both to '1' then every
     *                   single pixel is going to be evaluated.
     * @param oy         the Y-axis offset. See ox for details.
     * @param out        path of the output file (which corresponds to a colored img with right/wrong classified pixels)
     * @return ratio of corrected prediction over total evaluation
     * @throws IOException possible since it's writing a file on disk
     */
    private float getSingleClassError(DataBlock img, DataBlock gt, Classifier classifier, int ox, int oy, String out) throws IOException {
        int nbCorrect = 0;
        int nbWrong = 0;

        // Preparing output image: init the whole image to white
        BufferedImage res = null;
        if (printOutputFiles) {
            res = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
            int[] data = ((DataBufferInt) res.getRaster().getDataBuffer()).getData();
            Arrays.fill(data, 0xFFFFFF);
        }

        /* As we use x and y as centerInput() we need to leave half of the space left and right and we're fine because
         * we are sure that the top left corner of the patch will always be on the image. This also apply to the y axis
         * where we leave only half patch space top and bottom. Corner cases are handled correctly.
         */
        int index = gt.getDepth()-1;

        // Detect borders boundaries
        int xb = (classifier.getInputWidth() > ox) ? classifier.getInputWidth() / 2 : ox / 2;
        int yb = (classifier.getInputHeight() > oy) ? classifier.getInputHeight() / 2 : oy / 2;

        for (int x = xb; x < img.getWidth() - xb; x += ox) {
            for (int y = yb; y < img.getHeight() - yb; y += oy) {

                // Set input to classifier
                classifier.centerInput(img, x, y);

                // Forward
                classifier.compute();

                // Take the correct classification value from GT
                int correctClass = Math.round((gt.getValues(x, y)[index] + 1) * 255.0f / 2.0f);

                // Taking output class
                int outputClass = classifier.getOutputClass(false);

                // Error is computed simply comparing the two classes
                if (correctClass == outputClass) {
                    nbCorrect++;
                    if (printOutputFiles) {
                        for (int i = -(ox / 2); i < (ox / 2); i++) {
                            for (int j = -(oy / 2); j < (oy / 2); j++) {
                                res.setRGB(x + i, y + j, resultColor[1][1]);
                            }
                        }
                    }
                } else {
                    nbWrong++;
                    if (printOutputFiles) {
                        for (int i = -(ox / 2); i < (ox / 2); i++) {
                            for (int j = -(oy / 2); j < (oy / 2); j++) {
                                res.setRGB(x + i, y + j, resultColor[1][0]);
                            }
                        }
                    }
                }
            }
        }

        // Saving on disk the output image
        if (printOutputFiles) {
            ImageIO.write(res, "png", new File(out + ".png"));
        }

        return nbCorrect / (float) (nbCorrect + nbWrong);
    }


    /**
     * This methods evaluates the classifier expecting multi classes prediction. The cumulated error is computed as
     * sum of all miss classified classes. That is false negative + false positive are summed all together to compose
     * the final classification error. Each mistake counts as 1 regardless of "how bad" was the mistake. Optimally
     * the output images are all black.
     *
     * @param img        the image we will use as input
     * @param gt         the ground truth for the image provided as input
     * @param classifier the object classifier which has been selected to evaluate
     * @param ox         the X-axis offset which will be used to select the next pixel to be evaluated. In fact this methods
     *                   does not evaluate all the pixels but only one in every patch defined by ox and oy. This way it is
     *                   possible to execute a rough evaluation of the classifier without running every pixel of the imaged
     *                   trough the classification process (which would be slow). If ox and oy are set both to '1' then every
     *                   single pixel is going to be evaluated.
     * @param oy         the Y-axis offset. See ox for details.
     * @param out        path of the output file (which corresponds to a colored img with right/wrong classified pixels)
     * @return ratio of corrected prediction over total evaluation
     * @throws IOException possible since it's writing a file on disk
     */
    private float[] getMultipleClassesError(DataBlock img, DataBlock gt, Classifier classifier, int ox, int oy, String out) throws IOException {
        int nbClasses = classifier.getOutputSize();

        // Preparing output image
        BufferedImage[] res = new BufferedImage[nbClasses];
        if (printOutputFiles) {
            for (int i = 0; i < res.length; i++) {
                res[i] = new BufferedImage(gt.getWidth(), gt.getHeight(), BufferedImage.TYPE_INT_RGB);
                int[] data = ((DataBufferInt) res[i].getRaster().getDataBuffer()).getData();
                Arrays.fill(data, 0xFFFFFF);
            }
        }

        // Init error logging variables
        int[] truePositive = new int[nbClasses];
        int[] falsePositive = new int[nbClasses];
        int[] trueNegative = new int[nbClasses];
        int[] falseNegative = new int[nbClasses];

        /* As we use x and y as centerInput() we need to leave half of the space left and right and we're fine because
         * we are sure that the top left corner of the patch will always be on the image. This also apply to the y axis
         * where we leave only half patch space top and bottom. Corner cases are handled correctly.
         */
        int index = gt.getDepth()-1;

        // Detect borders boundaries
        int xb = (classifier.getInputWidth() > ox) ? classifier.getInputWidth() / 2 : ox / 2;
        int yb = (classifier.getInputHeight() > oy) ? classifier.getInputHeight() / 2 : oy / 2;

        for (int x = xb; x < img.getWidth() - xb; x += ox) {
            for (int y = yb; y < img.getHeight() - yb; y += oy) {

                // Set input to classifier
                classifier.centerInput(img, x, y);

                // Forward
                classifier.compute();

                // Take the correct classification value from GT
                int correctClass = Math.round((gt.getValues(x, y)[index] + 1) * 255 / 2.0f);
                // Convert int to bit-wise indicator. Example: 3(0011) -> 4th(1000)
                correctClass = 0x01 << correctClass;

                // Taking output class
                int outputClass = classifier.getOutputClass(true);

                // Error is computed
                for (int i = 0; i < nbClasses; i++) {
                    int r = (outputClass >> i) & 0x01;
                    int e = (correctClass >> i) & 0x01;

                    if (r == 0 && e == 0) {
                        trueNegative[i]++;
                    } else if ((r == 1 && e == 0)) {
                        falsePositive[i]++;
                    } else if ((r == 0 && e == 1)) {
                        falseNegative[i]++;
                    } else if ((r == 1 && e == 1)) {
                        truePositive[i]++;
                    }

                    if (printOutputFiles) {
                        for (int j = -(ox / 2); j < (ox / 2); j++) {
                            for (int k = -(oy / 2); k < (oy / 2); k++) {
                                res[i].setRGB(x + j, y + k, resultColor[r][e]);
                            }
                        }
                    }
                }
            }
        }

        // Saving on disk the output images
        if (printOutputFiles) {
            for (int i = 0; i < nbClasses; i++) {
                ImageIO.write(res[i], "png", new File(out + "-class-" + i + ".png"));
            }
        }

        /** Compute Precision, Recall and F1 score with the harmonic mean of precision and recall
         *  On the first position is the avg of all classes.         *
         */
        float[] recall = new float[nbClasses + 1];
        float[] precision = new float[nbClasses + 1];
        float[] f1 = new float[nbClasses + 1];
        int notPresentClass = 0;

        for (int i = 0; i < nbClasses; i++) {
            // Compute recall
            recall[i + 1] = truePositive[i] / ((float) truePositive[i] + falseNegative[i]);
            // If that class was not in the evaluated image recall will be NaN
            if (Float.isNaN(recall[i + 1])) {
                notPresentClass++;
                precision[i + 1] = Float.NaN;
                f1[i + 1] = Float.NaN;
                continue;
            }
            recall[0] += recall[i + 1];

            // Compute precision
            if (truePositive[i] + falsePositive[i] > 0) {
                precision[i + 1] = truePositive[i] / ((float) truePositive[i] + falsePositive[i]);
            }
            precision[0] += precision[i + 1];

            // Compute F1
            if (precision[i + 1] + recall[i + 1] > 0) {
                f1[i + 1] = 2 * ((precision[i + 1] * recall[i + 1]) / (precision[i + 1] + recall[i + 1]));
            }
            f1[0] += f1[i + 1];
        }
        precision[0] /= (nbClasses - notPresentClass);
        recall[0] /= (nbClasses - notPresentClass);
        f1[0] /= (nbClasses - notPresentClass);

        // Print results
        StringBuilder s = new StringBuilder();
        if (precision[0] == 0) {
            s.append(" [WARNING] No class has ever activated!");
            precision[0] = Float.NaN;
            recall[0] = Float.NaN;
        } else {
            s.append(" PRE=" + String.format("%.2f", precision[0]) + " [ ");
            for (int j = 1; j < precision.length; j++) {
                s.append(String.format("%.2f", precision[j]));
                if (j < precision.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]\t REC=" + String.format("%.2f", recall[0]) + " [ ");
            for (int j = 1; j < recall.length; j++) {
                s.append(String.format("%.2f", recall[j]));
                if (j < recall.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]\t F1=" + String.format("%.2f", f1[0]) + " [ ");
            for (int j = 1; j < f1.length; j++) {
                s.append(String.format("%.2f", f1[j]));
                if (j < f1.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]");
        }

        System.out.println(s.toString());

        return new float[]{precision[0], recall[0], f1[0]};

    }


    /**
     * This method is designed to evaluate a classifier with a dataset which has the GT on his filename.
     * Furthermore, it is assumed that each input image is same size as the classifier
     * input size. Examples of dataset like this are: CIFAR, MNIST.
     * I am sorry, I know this is not the most elegant way to implement but I lack
     * the time to do it better.
     * In the future maybe Dataset will contain the filename of the data it reads
     * and will make possible to merge this method with his Dataset based counterpart.
     * <p>
     * XML syntax to use this feature:
     * <p>
     * <evaluate-classifier ref="myClassifier">
     * <dataset>stringPATH</dataset>
     * <filenamebased/>
     * <subsampledataset>500</subsampledataset>    // Optional: specifies if the dataset should be subsampled
     * </evaluate-classifier>
     *
     * @param element the node of the XML directly
     */
    public String fileNameBased(Element element) throws IOException {

        /***********************************************************************************************
         * PARSE ELEMENT FROM XML
         **********************************************************************************************/

        // Fetching the classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        if (classifier == null) {
            error("cannot find classifier :" + ref);
        }

        // Fetching datasets - this must be a PATH
        File dsdir = new File(readElement(element, "dataset"));
        if (!dsdir.isDirectory()) {
            throw new RuntimeException("As there is the tag <filenamebased/>, <dataset> MUST contain a string with the path of the dataset (and not the reference of a dataset previously loaded)");
        }

        // Check whether dataset should be sub sampled
        int dsSubSample = 1;
        if (element.getChild("subsampledataset") != null) {
            dsSubSample = Integer.parseInt(readElement(element, "subsampledataset"));
        }

        // Parsing and verifying the output folder
        String outFile = null;
        if (element.getChild("output-folder") != null) {
            printOutputFiles = true;
            outFile = readElement(element, "output-folder");
        }

        /***********************************************************************************************
         * LOAD DATASET
         **********************************************************************************************/
        long startTime = System.currentTimeMillis();

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
         * EVALUATE CLASSIFIER
         **********************************************************************************************/

        // Init error logging variables
        int[] truePositive = new int[nbClasses];
        int[] falsePositive = new int[nbClasses];
        int[] trueNegative = new int[nbClasses];
        int[] falseNegative = new int[nbClasses];
        double[] accuracy = new double[nbClasses];

        // Init confusion matrix
        int[][] cm = new int[nbClasses][nbClasses];

        // Starting
        script.println("Start evaluating classifier: " + classifier.name());

        // Iterate over all classes
        for (int c = 0; c < nbClasses; c++) {

            // Convert int to bit-wise indicator. Example: 3(0011) -> 4th(1000)
            int correctClass = 0x01 << c;

            // Iterate over all images for that class
            for (int i = 0; i < data.get(c).size(); i++) {

                // Set input to classifier
                classifier.setInput(data.get(c).get(i), 0, 0);

                // Forward
                classifier.compute();

                // Taking output class
                int outputClass = classifier.getOutputClass(true);

                /* Update confusion matrix with ACCURACY instead of multiple-classes
                 * Columns are the labels, rows are the predictions.
                 * More info at: http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
                 */
                int o = classifier.getOutputClass(false);
                cm[o][c]++;

                // Increment the number of correct samples (used for accuracy)
                if(o==c){
                    accuracy[c]++;
                }

                // Error is computed
                for (int n = 0; n < nbClasses; n++) {
                    int r = (outputClass >> n) & 0x01;
                    int e = (correctClass >> n) & 0x01;

                    if (r == 0 && e == 0) {
                        trueNegative[n]++;
                    } else if ((r == 1 && e == 0)) {
                        falsePositive[n]++;
                    } else if ((r == 0 && e == 1)) {
                        falseNegative[n]++;
                    } else if ((r == 1 && e == 1)) {
                        truePositive[n]++;
                    }

                }
            }
            // Divide accuracy by total number of samples examined
            accuracy[c] /= data.get(c).size();
        }

        /***********************************************************************************************
         * COMPUTE ACCURACY
         **********************************************************************************************/

        // Average all classes accuracy
        double avgAcc = 0.0;
        for (int c = 0; c < nbClasses; c++) {
            avgAcc += accuracy[c]/nbClasses;
        }

        script.println("Single-class classifier evaluation : ACC=" + String.format("%.2f", avgAcc));

        /***********************************************************************************************
         * COMPUTE PRECISION, RECALL AND F1 SCORE
         **********************************************************************************************/

        /* F1 score with the harmonic mean of precision and recall
         * On the first position is the avg of all classes
         */
        float[] recall = new float[nbClasses + 1];
        float[] precision = new float[nbClasses + 1];
        float[] f1 = new float[nbClasses + 1];
        int notPresentClass = 0;

        for (int i = 0; i < nbClasses; i++) {
            // Compute recall
            recall[i + 1] = truePositive[i] / ((float) truePositive[i] + falseNegative[i]);
            // If that class was not in the evaluated image recall will be NaN
            if (Float.isNaN(recall[i + 1])) {
                notPresentClass++;
                precision[i + 1] = Float.NaN;
                f1[i + 1] = Float.NaN;
                continue;
            }
            recall[0] += recall[i + 1];

            // Compute precision
            if (truePositive[i] + falsePositive[i] > 0) {
                precision[i + 1] = truePositive[i] / ((float) truePositive[i] + falsePositive[i]);
            }
            precision[0] += precision[i + 1];

            // Compute F1
            if (precision[i + 1] + recall[i + 1] > 0) {
                f1[i + 1] = 2 * (precision[i + 1] * recall[i + 1] / (precision[i + 1] + recall[i + 1]));
            }
            f1[0] += f1[i + 1];
        }
        precision[0] /= (nbClasses - notPresentClass);
        recall[0] /= (nbClasses - notPresentClass);
        f1[0] /= (nbClasses - notPresentClass);

        // Print results
        StringBuilder s = new StringBuilder();
        s.append("Multiple-classes classifier evaluation: ");
        if (precision[0] == 0) {
            s.append("[WARNING] No class has ever activated!");
            precision[0] = Float.NaN;
            recall[0] = Float.NaN;
        } else {
            s.append(" PRE=" + String.format("%.2f", precision[0]) + " [ ");
            for (int j = 1; j < precision.length; j++) {
                s.append(String.format("%.2f", precision[j]));
                if (j < precision.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]\t REC=" + String.format("%.2f", recall[0]) + " [ ");
            for (int j = 1; j < recall.length; j++) {
                s.append(String.format("%.2f", recall[j]));
                if (j < recall.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]\t F1=" + String.format("%.2f", f1[0]) + " [ ");
            for (int j = 1; j < f1.length; j++) {
                s.append(String.format("%.2f", f1[j]));
                if (j < f1.length - 1) {
                    s.append(" | ");
                }
            }
            s.append(" ]");
        }

        System.out.println(s.toString());

        /***********************************************************************************************
         * SAVE CONFUSION MATRIX TO FILE
         **********************************************************************************************/

        if (printOutputFiles) {
            BufferedWriter outputWriter = new BufferedWriter(new FileWriter(outFile + ".txt"));
            for (int i = 0; i < cm.length; i++) {
                for (int j = 0; j < cm[i].length; j++) {
                    outputWriter.write(Integer.toString(cm[i][j]));
                    if (j + 1 < cm[i].length) {
                        outputWriter.write(",");
                    }
                }
                outputWriter.newLine();
            }
            outputWriter.flush();
            outputWriter.close();
        }

        // Free memory
        data = null;
        System.gc();

        return "";
    }

    @Override
    public String tagName() {
        return "evaluate-classifier";
    }
}
