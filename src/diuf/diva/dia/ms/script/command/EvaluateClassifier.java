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
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 * Evaluates the classification accuracy and stores the classification result.
 * @author Michele Alberti, Mathias Seuret
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
    private static final int[][] resultColor = {
            {0x000000, 0x0000FF},
            {0xFF0000, 0x00FF00}
    };

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public EvaluateClassifier(XMLScript script) {
        super(script);
    }

    /**
     * Evaluates a whole Dataset
     *
     * @param ds         Dataset to evaluate
     * @param classifier the classifier to evaluate
     * @param samples    number of images to evaluate in the dataset. This is useful for small/short evaluation
     * @return See elaborateResults() for more details.
     */
    public static double[][] evaluateDataSet(final Dataset<GroundTruthDataBlock> ds, final Classifier classifier, final int samples) {
        // TODO il 6 dovrebbe essewre il nunero di classi...
        double[][] averageResults = new double[4][6];
        for (int i = 0; i < ds.size(); i++) {
            double[][] result = evaluateDatablock(ds.get(i), classifier, samples / ds.size());
            // Integrate results
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < averageResults[0].length; k++) {
                    averageResults[j][k] += result[j][k] / ds.size();
                }
            }
        }

        return averageResults;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public-Static
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Evaluates a whole DataBlock
     * @param db DataBlock to evaluate
     * @param classifier the classifier to evaluate
     * @param ox         the X-axis offset which will be used to select the next pixel to be evaluated. In fact this methods
     *                   does not evaluate all the pixels but only one in every patch defined by ox and oy. This way it is
     *                   possible to execute a rough evaluation of the classifier without running every pixel of the imaged
     *                   trough the classification process (which would be slow). If ox and oy are set both to '1' then every
     *                   single pixel is going to be evaluated.
     * @param oy         the Y-axis offset. See ox for details.
     * @param outputPath path where the DataBlock will be printed on file. If null nothing will be saved.
     * @return accuracy, f1, precision, recall. See elaborateResults() for more details.
     */
    public static double[][] evaluateDatablock(final GroundTruthDataBlock db, final Classifier classifier, final int ox, final int oy, final String outputPath) {

        /*
         * Init the list. The size is computed as the number of times that both "for" executes. In other words,
         * it pre-computes the final value of 'i'. The +1 is because both x and y start at 0.
         */
        EvaluationSample[] eSamples = new EvaluationSample[(((db.getWidth() - classifier.getInputWidth()) / ox) + 1) * (((db.getHeight() - classifier.getInputHeight()) / oy) + 1)];

        // Prepare the sample list
        int i = 0;
        for (int x = classifier.getInputWidth() / 2; x < db.getWidth() - classifier.getInputWidth() / 2; x += ox) {
            for (int y = classifier.getInputHeight() / 2; y < db.getHeight() - classifier.getInputHeight()/2; y += oy) {
                // Take the correct classification value from GT
                eSamples[i] = new EvaluationSample(x, y, db, db.getGt(x, y));
                i++;
            }
        }

        // Evaluate them
        evaluateSamples(eSamples, classifier);

        // Print the Datablock
        paintDataBlock(eSamples, outputPath);

        // Print the confusion matrix
        confusionMatrix(eSamples, outputPath);

        return elaborateResults(eSamples);
    }

    /**
     * Evaluates a whole DataBlock
     *
     * @param db         DataBlock to evaluate
     * @param classifier the classifier to evaluate
     * @param samples    number of images to evaluate in the dataset. This is useful for small/short evaluation
     * @return accuracy, f1, precision, recall. See elaborateResults() for more details.
     */
    public static double[][] evaluateDatablock(final GroundTruthDataBlock db, final Classifier classifier, final int samples) {

        // Init the list
        EvaluationSample[] eSamples = new EvaluationSample[samples];

        // Prepare the sample list
        int i = 0;
        int c = -1;
        while (true) {
            c = (c + 1) % (db.getNumberOfClasses() + 1);
            Pixel p = db.getRandomRepresentative(c);
            // It means this class is not present in the current db
            if (p == null) continue;
            // If you get a NULL POINTER here, verify that you analyse() the Dataset!
            eSamples[i] = new EvaluationSample(p.x, p.y, db, db.getGt(p.x, p.y));
            i++;
            if (i == samples)
                break;
        }

        // Evaluate them
        evaluateSamples(eSamples, classifier);

        return elaborateResults(eSamples);
    }

    /**
     * Print the matrix returned from elaboratedResults() to screen
     *
     * @param results to be printed
     */
    public static String printResults(final double[][] results){
        int n = results[0].length;
        StringBuilder s = new StringBuilder();
        s.append(" ACC=" + String.format("%.2f", results[0][0]) + " [ ");
        for (int i = 1; i < n; i++) {
            s.append(String.format("%.2f", results[0][i]));
            if (i < n - 1) s.append(" | ");
        }
        s.append(" ]");
        if (results[2][0] == 0) {
            s.append("[WARNING] No class has ever activated!");
            results[2][0] = Float.NaN;
            results[3][0] = Float.NaN;
        } else {
            s.append("\t F1=" + String.format("%.2f", results[1][0]) + " [ ");
            for (int i = 1; i < n; i++) {
                s.append(String.format("%.2f", results[1][i]));
                if (i < n - 1) s.append(" | ");
            }
            s.append(" ]\t PRE=" + String.format("%.2f", results[2][0]) + " [ ");
            for (int i = 1; i < n; i++) {
                s.append(String.format("%.2f", results[2][i]));
                if (i < n - 1) s.append(" | ");
            }
            s.append(" ]\t REC=" + String.format("%.2f", results[3][0]) + " [ ");
            for (int i = 1; i < n; i++) {
                s.append(String.format("%.2f", results[3][i]));
                if (i < n-1) s.append(" | ");
            }
            s.append(" ]");
        }

        return s.toString();

    }

    /**
     * This method is the core of the evaluation. It takes evaluations samples and run them.
     * @param eSamples the samples to be evaluated
     * @param classifier the classifier to evaluate
     * @return the samples evaluated
     */
    private static void evaluateSamples(final EvaluationSample[] eSamples, final Classifier classifier) {

        // Iterate over all samples
        for (EvaluationSample e : eSamples) {

            // Set input to classifier
            classifier.centerInput(e.db, e.x, e.y);

            // Forward
            classifier.compute();

            // Take MULTI CLASS response
            e.m = classifier.getOutputClass(true);

            // Take SINGLE CLASS response
            e.s = classifier.getOutputClass(false);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Private-Static
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method elaborates the samples and provides accuracy, F1, precision and recall
     *
     * @param eSamples the samples to be elaborated
     * @return row-wise are accuracy, f1, precision, recall. In this order.
     * On the left-most columns (index 0) is the avg, then follows all class-wise values.
     * It is structured as follows:
     * [ avg ACC, ACC class1, ACC class2, ... ACC class n ;
     * avg F1,  F1  class1, F1  class2, ... F1  class n ;
     * avg PRE, PRE class1, PRE class2, ... PRE class n ;
     * avg REC, REC class1, REC class2, ... REC class n ]
     */
    private static double[][] elaborateResults(final EvaluationSample[] eSamples) {
        final int nbClasses = getNbClasses(eSamples);

        final int[] classesSize = new int[nbClasses];

        // Init the return value matrix
        final double[][] rv = new double[4][nbClasses + 1];

        // Init error logging variables
        final int[] truePositive = new int[nbClasses];
        final int[] falsePositive = new int[nbClasses];
        final int[] trueNegative = new int[nbClasses];
        final int[] falseNegative = new int[nbClasses];
        final double[] accuracy = new double[nbClasses];

        // Iterate over all samples
        for (EvaluationSample s : eSamples) {

            // Update the classes size accordingly such that in the end we know how many of each class we evaluated
            classesSize[s.gt]++;

            // Convert int to bit-wise indicator. Example: 3(0011) -> 4th(1000)
            int correctClass = 0x01 << s.gt;

            // Error is computed
            for (int n = 0; n < nbClasses; n++) {
                int r = (s.m >> n) & 0x01;
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

            // Increment the number of correct samples (used for accuracy)
            if (s.s == s.gt) {
                accuracy[s.gt]++;
            }
        }

        /***********************************************************************************************
         * COMPUTE ACCURACY
         **********************************************************************************************/
        for (int i = 0; i < nbClasses; i++) {
            // Divide accuracy by number of samples examined of that class
            rv[0][i + 1] = accuracy[i] / classesSize[i];
            // Integrate accuracy, then we will divide to obtain weighted AVG (class balanced)
            rv[0][0] += accuracy[i];
        }
        rv[0][0] /= eSamples.length;

        /***********************************************************************************************
         * COMPUTE PRECISION, RECALL AND F1 SCORE
         **********************************************************************************************/
        int notPresentClass = 0;

        for (int i = 0; i < nbClasses; i++) {
            // Compute recall
            rv[3][i + 1] = truePositive[i] / ((float) truePositive[i] + falseNegative[i]);
            // If that class was not in the evaluated image recall will be NaN
            if (Double.isNaN(rv[3][i + 1])) {
                notPresentClass++;
                rv[2][i + 1] = Float.NaN;
                rv[1][i + 1] = Float.NaN;
                continue;
            }
            rv[3][0] += rv[3][i + 1];

            // Compute precision
            if (truePositive[i] + falsePositive[i] > 0) {
                rv[2][i + 1] = truePositive[i] / ((float) truePositive[i] + falsePositive[i]);
            }
            rv[2][0] += rv[2][i + 1];

            // Compute F1
            if (rv[2][i + 1] + rv[3][i + 1] > 0) {
                rv[1][i + 1] = 2 * (rv[2][i + 1] * rv[3][i + 1] / (rv[2][i + 1] + rv[3][i + 1]));
            }
            rv[1][0] += rv[1][i + 1];
        }
        rv[2][0] /= (nbClasses - notPresentClass);
        rv[3][0] /= (nbClasses - notPresentClass);
        rv[1][0] /= (nbClasses - notPresentClass);

        return rv;
    }

    /**
     * Visualize the results of the evaluation and saves them on file
     *
     * @param eSamples   samples with the result
     * @param outputPath path where to save the files
     */
    private static void paintDataBlock(final EvaluationSample[] eSamples, final String outputPath) {

        if (outputPath == null) {
            return;
        }

        // Determine number of classes
        final int nbClasses = getNbClasses(eSamples);

        // Reference to the initial DataBlock
        final DataBlock db = eSamples[0].db;

        // Determine offsets
        int k = 0;
        while (eSamples[k].x == eSamples[k + 1].x) k++;
        final int ox = eSamples[k + 1].x - eSamples[k].x;
        k = 0;
        while (eSamples[k].y == eSamples[k + 1].y) k++;
        final int oy = eSamples[k + 1].y - eSamples[k].y;

        // Preparing output image: init the whole image to white (it is useful to spot mistakes on borders or offset)
        BufferedImage resSingle = new BufferedImage(db.getWidth(), db.getHeight(), BufferedImage.TYPE_INT_RGB);
        int[] data = ((DataBufferInt) resSingle.getRaster().getDataBuffer()).getData();
        Arrays.fill(data, 0xFFFFFF);

        BufferedImage[] resMulti = new BufferedImage[nbClasses];
        for (int i = 0; i < resMulti.length; i++) {
            resMulti[i] = new BufferedImage(db.getWidth(), db.getHeight(), BufferedImage.TYPE_INT_RGB);
            data = ((DataBufferInt) resMulti[i].getRaster().getDataBuffer()).getData();
            Arrays.fill(data, 0xFFFFFF);
        }

        for (EvaluationSample s : eSamples) {
            if (s.gt == s.s) {
                for (int i = 0; i <= ox && s.x + i < resSingle.getWidth(); i++) {
                    for (int j = 0; j <= oy && s.y + j < resSingle.getHeight(); j++) {
                        resSingle.setRGB(s.x + i, s.y + j, resultColor[1][1]);
                    }
                }
            } else {
                for (int i = 0; i <= ox && s.x + i < resSingle.getWidth(); i++) {
                    for (int j = 0; j <= oy && s.y + j < resSingle.getHeight(); j++) {
                        resSingle.setRGB(s.x + i, s.y + j, resultColor[1][0]);
                    }
                }
            }

            // Error is computed
            for (int c = 0; c < nbClasses; c++) {
                int r = (s.m >> c) & 0x01;
                int e = ((0x01 << s.gt) >> c) & 0x01;

                for (int i = 0; i <= ox && s.x + i < resSingle.getWidth(); i++) {
                    for (int j = 0; j <= oy && s.y + j < resSingle.getHeight(); j++) {
                        resMulti[c].setRGB(s.x + i, s.y + j, resultColor[r][e]);
                    }
                }
            }
        }

        // Saving on disk the output image
        int i = 0;
        File file;
        // Looks for next free number on the output path folder and the saves in disk
        while ((file = new File(outputPath + i + "-eval-accuracy.png")).exists()) i++;

        try {
            ImageIO.write(resSingle, "png", file);
            for (int c = 0; c < nbClasses; c++) {
                ImageIO.write(resMulti[c], "png", new File(outputPath + i + "-eval-class-" + c + ".png"));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    /**
     * This methods computes the confusion matrix given a list of evaluated samples.
     *
     * @param eSamples list of evaluated samples
     * @param out      the path where the confusion matrix will be printed. Can be null.
     * @return the confusion matrix
     */
    private static int[][] confusionMatrix(final EvaluationSample[] eSamples, String out) {

        final int nbClasses = getNbClasses(eSamples);

        // Init confusion matrix
        int[][] cm = new int[nbClasses][nbClasses];

        /* Update confusion matrix with ACCURACY instead of multiple-classes
         * Columns are the labels, rows are the predictions.
         * More info at: http://text-analytics101.rxnlp.com/2014/10/computing-precision-and-recall-for.html
         */
        for (EvaluationSample e : eSamples) {
            cm[e.s][e.gt]++;
        }

        if (out != null) {
            // Save on file original CM
            saveMatrixOnFile(cm, out + "cm.txt");

            // Scale it in percentage
            int[] row = new int[nbClasses];
            for (int i = 0; i < nbClasses; i++) {
                for (int j = 0; j < nbClasses; j++) {
                    row[i] += cm[i][j];
                }
                for (int j = 0; j < nbClasses; j++) {
                    if (row[i] != 0) cm[i][j] = (cm[i][j] * 100) / row[i];
                }
            }

            // Save on file Scaled
            saveMatrixOnFile(cm, out + "cm-scaled.txt");
        }

        return cm;
    }

    /**
     * Saves a matrix to file
     *
     * @param m       the matrix to save
     * @param outPath the location where to save it
     */
    private static void saveMatrixOnFile(int[][] m, String outPath) {
        try {
            BufferedWriter outputWriter = new BufferedWriter(new FileWriter(outPath));
            for (int i = 0; i < m.length; i++) {
                for (int j = 0; j < m[i].length; j++) {
                    outputWriter.write(Integer.toString(m[i][j]));
                    if (j + 1 < m[i].length) {
                        outputWriter.write(",");
                    }
                }
                outputWriter.newLine();
            }
            outputWriter.flush();
            outputWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * @param eSamples the list of samples
     * @return the number of classes
     */
    private static int getNbClasses(EvaluationSample[] eSamples) {
        int nbClasses = 0;
        for (EvaluationSample e : eSamples) {
            nbClasses = Math.max(e.gt, nbClasses);
        }
        // The class 0 is a class and needs to be added
        nbClasses++;
        return nbClasses;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // XML
    ///////////////////////////////////////////////////////////////////////////////////////////////
    @Override
    public String execute(Element element) throws Exception {

        // Fetch the classifier
        String ref = readAttribute(element, "ref");
        Classifier classifier = script.classifiers.get(ref);
        if (classifier == null) {
            error("cannot find classifier :" + ref);
        }

        // Fetch dataset
        String dataset = readElement(element, "dataset");
        Dataset<GroundTruthDataBlock> ds = script.datasets.get(dataset);
        if (ds == null) {
            throw new RuntimeException("Dataset: " + dataset + " not found!");
        }

        // Parsing the offsets
        int offsetX = (element.getChild("offset-x") != null) ? Integer.parseInt(readElement(element, "offset-x")) : 1;
        int offsetY = (element.getChild("offset-y") != null) ? Integer.parseInt(readElement(element, "offset-y")) : 1;

        // Parsing and verifying the output folder
        String outPath = null;
        if (element.getChild("output-folder") != null) {
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
        XMLScript.println("Start evaluating classifier: " + classifier.name());

        double[][] averageResults = new double[4][classifier.getOutputSize() + 1];

        for (int i = 0; i < ds.size(); i++) {
            double[][] result = evaluateDatablock(ds.get(i), classifier, offsetX, offsetY, outPath);
            XMLScript.println("Classified image " + (i + 1) + "/" + ds.size() + " : " + printResults(result));
            // Integrate results
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < averageResults[0].length; k++) {
                    averageResults[j][k] += result[j][k] / ds.size();
                }
            }
        }

        // Print results over all images
        XMLScript.println("Finish evaluating classifier: " + printResults(averageResults));

        return String.valueOf(averageResults[0][0]);
    }

    @Override
    public String tagName() {
        return "evaluate-classifier";
    }

    /**
     * This class stores a pixel coordinate and their respective datablock.
     * In case of a feature based ds, the pixel should be 0,0.
     * <p>
     * This class is insecure because you can potentially fetch uninitialized values of s and m.
     * Make it secure would make it slow with 'if' construct which would be executed many times.
     * Be safe, think about what you are doing!
     */
    private static class EvaluationSample {
        /**
         * Pixel coordinates
         */
        public final int x, y;
        /**
         * Datablock where the pixel belong to
         */
        public final DataBlock db;
        /**
         * Gt of the pixel. Note that its an int!
         */
        public final int gt;
        /**
         * The result of the evaluation, on single class
         */
        public int s;
        /**
         * The result of the evaluation, multi classes
         */
        public int m;

        public EvaluationSample(final Pixel p, final DataBlock db, final int gt) {
            this(p.x, p.y, db, gt);
        }

        public EvaluationSample(final int x, final int y, final DataBlock db, final int gt) {
            this.x = x;
            this.y = y;
            this.db = db;
            this.gt = gt;
        }
    }
}
