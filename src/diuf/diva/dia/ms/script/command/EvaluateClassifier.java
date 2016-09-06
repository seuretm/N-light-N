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
import org.jdom2.Element;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * Evaluates the classification accuracy and stores the classification result.
 * Syntax for the XML command:
 *
 *  <evaluate-classifier ref="">
 *      <dataset>stringID</dataset>                             // ID of the dataset to be tested
 *      <groundTruth>stringID</groundTruth>                     // ID of the dataset ground truth
 *      <offset-x>int</offset-x>                                // Not all pixel are going to be evaluated, this specifies the x-offset
 *      <offset-y>int</offset-y>                                // Not all pixel are going to be evaluated, this specifies the y-offset
 *      <method>enum(single-class,multiple-classes)</method>    // Specifies whether the error should be computed single or multi class*
 *      <output-folder>stringPATH</output-folder>               // Path of the output folder
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

    public EvaluateClassifier(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {

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
        String outPath = readElement(element, "output-folder");
        File file = new File(outPath);
        if (!file.exists()) {
            file.mkdirs();
        }
        if (!file.isDirectory()) {
            error(outPath + " is not a folder");
        }

        // Starting
        script.println("Start evaluating classifier: " + classifier.name());

        float[] cumulatedError = {0, 0};
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
                    break;
            }
        }

        // Print results over all images
        switch (et) {
            case SINGLE_CLASS:
                script.println("Finish evaluating classifier single-class: ACC=" + String.format("%.2f", cumulatedError[0] / ds.size()));
                break;
            case MULTIPLE_CLASSES:
                script.println("Finish evaluating classifier multiple-classes: PRE=" + String.format("%.2f", cumulatedError[0] / ds.size()) + " REC=" + String.format("%.2f", cumulatedError[1] / ds.size()));
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
        int patchSizeX = classifier.getInputWidth();
        int patchSizeY = classifier.getInputHeight();
        int nbCorrect = 0;
        int nbWrong = 0;

        // Preparing output image: init the whole image to white
        BufferedImage res = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_INT_RGB);
        int[] data = ((DataBufferInt) res.getRaster().getDataBuffer()).getData();
        Arrays.fill(data, 0xFFFFFF);

        /* As we use x and y as centerInput() we need to leave half of the space left and right and we're fine because
         * we are sure that the top left corner of the patch will always be on the image. This also apply to the y axis
         * where we leave only half patch space top and bottom. Corner cases are handled correctly.
         */
        int index = gt.getDepth()-1;
        int nbNonNull = 0;
        for (int x = patchSizeX / 2; x < img.getWidth() - patchSizeX / 2; x += ox) {
            for (int y = patchSizeY / 2; y < img.getHeight() - patchSizeY / 2; y += oy) {

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
                    for (int i = 0; i < ox; i++) {
                        for (int j = 0; j < oy; j++) {
                            res.setRGB(x + i, y + j, resultColor[1][1]);
                        }
                    }
                } else {
                    // Debugging
                    //System.out.println("o=" + outputClass + ",c=" + correctClass);
                    //((FFCNN)classifier).getOutputScores();
                    nbWrong++;
                    for (int i = 0; i < ox; i++) {
                        for (int j = 0; j < oy; j++) {
                            res.setRGB(x + i, y + j, resultColor[1][0]);
                        }
                    }
                }
            }
        }

        // Saving on disk the output image
        ImageIO.write(res, "png", new File(out + ".png"));

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
        int patchSizeX = classifier.getInputWidth();
        int patchSizeY = classifier.getInputHeight();
        int nbClasses = classifier.getOutputSize();

        // Preparing output image
        BufferedImage[] res = new BufferedImage[nbClasses];
        for (int i = 0; i < res.length; i++) {
            res[i] = new BufferedImage(gt.getWidth(), gt.getHeight(), BufferedImage.TYPE_INT_RGB);
            int[] data = ((DataBufferInt) res[i].getRaster().getDataBuffer()).getData();
            Arrays.fill(data, 0xFFFFFF);
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
        for (int x = patchSizeX / 2; x < img.getWidth() - patchSizeX / 2; x += ox) {
            for (int y = patchSizeY / 2; y < img.getHeight() - patchSizeY / 2; y += oy) {

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

                    // Debugging
                    //System.out.println("e=" + e + ",r=" + r + ",oc=" + outputClass);
                    //((FFCNN)classifier).getOutputScores();

                    for (int j = 0; j < ox; j++) {
                        for (int k = 0; k < oy; k++) {
                            res[i].setRGB(x + j, y + k, resultColor[r][e]);
                        }
                    }
                }
            }
        }

        // Saving on disk the output images
        for (int i = 0; i < nbClasses; i++) {
            ImageIO.write(res[i], "png", new File(out + "-class-" + i + ".png"));
        }

        // Compute precision (on the first position is the avg of all classes)
        float[] precision = new float[nbClasses + 1];
        for (int i = 0; i < nbClasses; i++) {
            if (truePositive[i] + falsePositive[i] == 0) {
                precision[i + 1] = 0;
            } else {
                precision[i + 1] = truePositive[i] / ((float) truePositive[i] + falsePositive[i]);
            }
            precision[0] += precision[i + 1];
        }
        precision[0] /= nbClasses;

        // Compute recall (on the first position is the avg of all classes)
        float[] recall = new float[nbClasses + 1];
        int notPresentClass = 0;
        for (int i = 0; i < nbClasses; i++) {
            recall[i + 1] = truePositive[i] / ((float) truePositive[i] + falseNegative[i]);
            // Only add it if is not NaN, which could be produced if that class was not in the evaluated image
            if (recall[i + 1] == recall[i + 1]) {
                recall[0] += recall[i + 1];
            } else {
                notPresentClass++;
            }
        }
        recall[0] /= (nbClasses - notPresentClass);

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
            s.append(" ]");
        }

        System.out.println(s.toString());

        return new float[]{precision[0], recall[0]};

    }

    @Override
    public String tagName() {
        return "evaluate-classifier";
    }
}
