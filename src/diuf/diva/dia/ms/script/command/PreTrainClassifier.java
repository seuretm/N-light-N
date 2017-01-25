package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.PreTrainable;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.BiDataBlock;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.misc.ImageAnalysis;
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * This class PRE-trains a classifier previously loaded into the script. It is very similar to TrainClassifier
 * but with the difference that only a subset of classifier (those who implement PreTrainable) can be trained
 * this way. Typically FFCNN would be pre-trained when using the optional paramter <ae>LDA</ae>
 *
 * @author Michele Alberti
 */
public class PreTrainClassifier extends AbstractCommand {

    public PreTrainClassifier(XMLScript script) {
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

        if (!(classifier instanceof PreTrainable)) {
            // If the classifier is not a PreTrainable then abort
            throw new ClassCastException("This classifier does not implement PreTrainable.java -> " +
                    "hence pre training is not supported");
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

        // Train the classifier
        script.println("\"SCAE Starting PRE-training[" + ref + "] {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "}");

        long startTime = System.currentTimeMillis();


        switch (classifier.type()) {
            case "pixel":
                preTrainPixelBasedClassifier(classifier, ds, gt);
                break;

            default:
                error("invalid classifier type :" + classifier.type());
        }

        script.println("PRE-Training time = " + (int) (System.currentTimeMillis() - startTime) / 1000.0);
        script.println("Finish PRE-training classifier [" + ref + "]");

        return "";
    }

    /**
     * This method is designed to pre-train a classifier with pixel-based dat as input.
     *
     * XML syntax to use this feature:
     *
     * <pre-train-classifier ref="myClassifier">
     * <dataset>stringID</dataset>           // ID of the testing data set
     * <groundTruth>stringID</groundTruth>   // ID of the ground truth data set
     * <samples>int</samples>
     * <max-time>int</max-time>
     * </pre-train-classifier>
     *
     * @param classifier the classifier which is going to be trained
     * @param dsImg      the dataset containing the images which will be used for training
     * @param dsGt       the dataset containing the ground truth for the provided dataset
     *                   is immediately interrupted and the result returned
     */
    private void preTrainPixelBasedClassifier(Classifier classifier, Dataset dsImg, Dataset dsGt) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Counter that keeps track on how many samples have been already executed
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

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

                    // Set input to classifier
                    ((PreTrainable) classifier).addTrainingSample(c);

                    // Increase counters
                    sample++;

                    // Stop execution if MAXTIME reached
                    if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                        // Complete the logging progress
                        System.out.println("]");
                        script.println("Maximum PRE-training time (" + MAXTIME + ") reached after " + sample + " samples");
                        ((PreTrainable) classifier).trainingDone();
                        return;
                    }
                }
            }
        }

        // Complete the logging progress
        System.out.println(" 100%]");

        ((PreTrainable) classifier).trainingDone();
    }

    /**
     * This method is designed to pre-train a classifier with the GT on his filename.
     * Furthermore, it is assumed that each input image is same size as the classifier
     * input size. Examples of dataset like this are: CIFAR, MNIST.
     * I am sorry, I know this is not the most elegant way to implement but I lack
     * the time to do it better.
     * In the future maybe Dataset will contain the filename of the data it reads
     * and will make possible to merge this method with his Dataset based counterpart.
     * <p>
     * XML syntax to use this feature:
     * <p>
     * <pre-train-classifier ref="myClassifier">
     * <dataset>stringPATH</dataset>           // PATH (and not ID!) of the training data set
     * </filenamebased>
     * <subsampledataset>500</subsampledataset>    // Optional: specifies if the dataset should be subsampled
     * <samples>int</samples>
     * <max-time>int</max-time>
     * </pre-train-classifier>
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
        startTime = System.currentTimeMillis();

        script.println("\"Classifier starting PRE-training[" + ref + "] {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "}");

        script.print("Progress[");

        // Train the classifier until enough samples have been evaluated
        while (sample < SAMPLES) {

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
                classifier.setInput(data.get(c).get((int) (Math.random() * data.get(c).size())), 0, 0);

                // Forward
                classifier.compute();

                // Set input to classifier
                ((PreTrainable) classifier).addTrainingSample(c);

                // Increase counters
                sample++;

                // Stop execution if MAXTIME reached
                if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                    // Complete the logging progress
                    System.out.println("]");
                    script.println("Maximum training time (" + MAXTIME + ") reached after " + sample + " samples");

                    // Free memory
                    data = null;
                    System.gc();

                    return;
                }
            }
        }

        // Complete the logging progress
        System.out.println(" 100%]");

        script.println("Training time = " + (int) (System.currentTimeMillis() - startTime) / 1000.0);
        script.println("Finish training classifier [" + ref + "]");

        // Free memory
        data = null;
        System.gc();
    }

    @Override
    public String tagName() {
        return "pre-train-classifier";
    }

}
