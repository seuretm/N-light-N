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

import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.*;
import diuf.diva.dia.ms.util.misc.ImageAnalysis;
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;

/**
 * Trains an autoencoder or a denoising autoencoder. It allows to display and save on file a plot of the training
 * error.
 *
 * XML syntax to use this feature:
 *
 *  <train-scae ref="myScae">
 *      <dataset>ds</dataset>
 *      <groundTruth>gt</groundTruth>                       // optional: only for supervised AE
 *      <samples>SAMPLES</samples>
 *      <max-time>MAXTIME</max-time>
 *      <!-- optional -->
 *      <display-features>200</display-features>
 *      <!-- optional -->
 *      <display-recoding>stringPATH</display-recoding>
 *      <!-- optional -->
 *      <display-progress>200</display-progress>
 *      <!-- optional, but needs display-progress -->
 *      <save-progress>stringPATH</save-progress>
 *  </train-scae>
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class TrainSCAE extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public TrainSCAE(XMLScript script) {
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
     * Support variables for the <display-features> tag
     */
    int tracerFeaturesUpdateStep = 1000;
    int currTracerFeatures = 0;

    @Override
    public String execute(Element element) throws Exception {

        // Get the SCAE
        String ref = readAttribute(element, "ref");
        SCAE scae = script.scae.get(ref);
        if (scae==null) {
            error("cannot find "+ref+", check the id");
        }

        assert (scae != null);

        // Parse the dataset
        String dataset = readElement(element, "dataset");
        Dataset ds = script.datasets.get(dataset);
        NoisyDataset nds = script.noisyDataSets.get(dataset);

        if (ds == null && nds == null && element.getChild("filenamebased") == null) {
            error("cannot find dataset " + dataset);
        }

        // Parse the (optional, used only in supervised) ground truth dataset
        Dataset gt = null;
        if (element.getChild("groundTruth") != null) {
            gt = script.datasets.get(readElement(element, "groundTruth"));
        }

        // Parse parameter for training
        this.SAMPLES = Integer.parseInt(readElement(element, "samples"));
        this.MAXTIME = Integer.parseInt(readElement(element, "max-time"));

        if (SAMPLES == 0 && MAXTIME == 0) {
            error("cannot set both epochs and max-time to 0");
        }
        if (SAMPLES == 0) {
            MAXTIME = Integer.MAX_VALUE;
        }

        if (MAXTIME == 0) {
            MAXTIME = Integer.MAX_VALUE;
        }

        // If display-progress is present, init the tracer
        tracer = null;
        if (element.getChild("save-progress") != null) {
            try {

                /* Expected points on the plots are the number of expected epochs.
                 * So number of total samples divided by size of epoch
                 */
                int expectedSamples;
                if (ds != null) {
                    expectedSamples = SAMPLES / ds.size();
                } else {
                    expectedSamples = SAMPLES / nds.clean.size();
                }
                tracer = new Tracer(
                        "SCAE training error",
                        "Epoch",
                        "Error",
                        expectedSamples,
                        element.getChild("display-progress") != null
                );
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // If display-feature tag is present init it
        FeatureDisplay featureDisplay = null;
        if (element.getChild("display-features")!=null) {
            featureDisplay = new FeatureDisplay(scae);
            try {
                tracerFeaturesUpdateStep = Integer.parseInt(element.getChildText("display-features"));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        // If recoding tag is present init it
        RecodingDisplay recodingDisplay = null;
        if (element.getChild("display-recoding")!=null) {
            Image img = new Image(element.getChild("display-recoding").getTextTrim());
            img.convertTo(script.colorspace);
            DataBlock imgDB = new DataBlock(img);
            recodingDisplay = new RecodingDisplay(imgDB, scae);
        }

        // Return value of the function
        String returnValue = null;

        // Query the normal datasets
        if (ds!=null) {
            // Verify dataset size before training
            for (DataBlock db : ds) {
                if (db.getWidth() < scae.getInputPatchWidth() || db.getHeight() < scae.getInputPatchHeight()) {
                    error(
                            "an image of the dataset is smaller than the input of the autoencoder:\n"
                                    + "["
                                    + db.getWidth()
                                    + "x"
                                    + db.getHeight()
                                    + "]<["
                                    + scae.getInputPatchWidth()
                                    + "x"
                                    + scae.getInputPatchHeight()
                                    + "]"
                    );
                }
            }

            if (gt != null) {
                returnValue = String.valueOf(trainSupervisedAutoEncoder(scae, ds, gt, featureDisplay, recodingDisplay));
            } else {
                returnValue = String.valueOf(trainAutoencoder(scae, ds, featureDisplay, recodingDisplay));
            }
        }

        // Query the noisy dataset
        if (nds!=null) {
            // Verify dataset size before training (on clean only is sufficient!)
            for (DataBlock db : nds.clean) {
                if (db.getWidth() < scae.getInputPatchWidth() || db.getHeight() < scae.getInputPatchHeight()) {
                    error(
                            "an image of the dataset is smaller than the input of the autoencoder:\n"
                                    + "["
                                    + db.getWidth()
                                    + "x"
                                    + db.getHeight()
                                    + "] < ["
                                    + scae.getInputPatchWidth()
                                    + "x"
                                    + scae.getInputPatchHeight()
                                    + "]"
                    );
                }
            }
            returnValue = String.valueOf(trainDenoisingAutoEncoder(scae, nds, featureDisplay, recodingDisplay));
        }

        /** If the tag </filenamebased> is present
         *  call the relative method. Otherwise go on
         *  with typical training.
         */
        if (element.getChild("filenamebased") != null) {
            returnValue = String.valueOf(trainSupervisedFileNameBasedAutoEncoder(scae, element, featureDisplay, recodingDisplay));
        }

        // If a result has been computed
        if (returnValue != null) {
            // If tracer has been used
            if (tracer != null) {
                tracer.addRawData();
                tracer.addCumulatedAverage();
                tracer.addMovingAverage();
                tracer.addMovingMedian();
                // Show the plot
                tracer.display();

                // If requires, save the plot on disk at the provided path
                if (element.getChild("save-progress") != null) {
                    try {
                        tracer.savePlot(readElement(element, "save-progress"));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
            return returnValue;
        }

        return "";
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // AUTO ENCODER
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Train an auto encoder
     *
     * @param scae the scae to be trained
     * @param ds   the data set to train that will be used to train the scae
     * @param fd   the feature display object (may be null!)
     * @param rd   the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainAutoencoder(SCAE scae, Dataset ds, FeatureDisplay fd, RecodingDisplay rd) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Error epoch wise
        double err;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Cumulated training error
        double cumulatedError = 0;

        // Number of samples already evaluated
        int sample = 1;

        // Epoch counter
        int epoch = 0;

        // Random numbers generator
        Random rand = XMLScript.getRandom();

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Iterate until enough samples has been evaluated
        while (sample <= SAMPLES) {

            // Shuffle the dataset at each epoch
            ds.randomPermutation();

            // Epoch-wise error
            err = 0;

            // Log every ~10% the progress of training
            if (((sample - 1) * 10) / SAMPLES >= loggingProgress) {
                System.out.print(loggingProgress * 10 + "% ");
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
                loggingProgress = (sample * 10) / SAMPLES + 1;

            }

            // At each epoch we iterate over all images in the dataset
            for (DataBlock db : ds) {

                int x = 0;
                int y = 0;

                if (db.getWidth() - scae.getInputPatchWidth() > 0) {
                    x = rand.nextInt(db.getWidth() - scae.getInputPatchWidth());
                }
                if (db.getHeight()>scae.getInputPatchHeight()) {
                    y = rand.nextInt(db.getHeight() - scae.getInputPatchHeight());
                }

                // Set input
                scae.setInput(db, x, y);


                err += scae.train();


                // Increase counter of examined samples
                sample++;

                currTracerFeatures++;
            }

            // Add the new epoch point to the plot
            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err);
            }

            // Feature display update
            if (fd!=null && currTracerFeatures>=tracerFeaturesUpdateStep) {
                fd.update();
            }

            // Recoding update
            if (rd!=null) {
                rd.update();
            }

            // Log the number of epochs
            epoch++;

            // Update the cumulated error
            cumulatedError += err;

            // Stop execution if MAXTIME reached
            if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                // Complete the logging progress
                System.out.println("]");
                script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                scae.trainingDone();
                return cumulatedError;
            }

        }

        // Complete the logging progress
        if (sample >= SAMPLES) {
            System.out.println("100%]");
        }

        scae.trainingDone();

        return cumulatedError;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // SUPERVISED AUTO ENCODER
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Train an auto encoder which requires a supervised training type.
     * For example, LDAAutoEncoder needs class labels for his initial training.
     *
     * @param scae the scae to be trained
     * @param dsImg the dataset to train that will be used to train the scae
     * @param dsGt the dataset ground truth used to extract the labels of training data
     * @param fd the feature display object (may be null!)
     * @param rd the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainSupervisedAutoEncoder(SCAE scae, Dataset dsImg, Dataset dsGt, FeatureDisplay fd, RecodingDisplay rd) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Number of samples already evaluated
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Counter of epochs (logging purpose only)
        int epoch = 0;

        // Epoch-wise error
        double err;
        int epochSize;

        // Cumulated training error
        double cumulatedError = 0;

        // ImageAnalysis init, for data balancing
        ImageAnalysis[] imageAnalyses = new ImageAnalysis[dsImg.size()];
        for (int i = 0; i < dsImg.size(); i++) {
            imageAnalyses[i] = new ImageAnalysis(dsGt.get(i), scae.getInputPatchWidth(), scae.getInputPatchHeight());
            imageAnalyses[i].subSample((int) Math.ceil(SAMPLES / dsImg.size()));
        }

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Iterate until enough samples has been evaluated
        while (sample < SAMPLES) {

            // Epoch-wise error
            err = 0;
            epochSize = 0;

            // Log every ~10% the progress of training
            if (((sample - 1) * 10) / SAMPLES >= loggingProgress) {
                System.out.print(loggingProgress * 10 + "% ");
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
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

                    // Set input
                    scae.centerInput(dsImg.get(i), p.x, p.y);

                    // Train
                    err += scae.trainSupervised(c);

                    // Increase counters
                    sample++;
                    epochSize++;

                    // Stop execution if MAXTIME reached
                    if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                        // Complete the logging progress
                        System.out.println("]");
                        script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                        scae.trainingDone();
                        return cumulatedError;
                    }
                }
            }

            // Add the new epoch point to the plot
            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err / epochSize);
            }

            // Feature display update
            if (fd != null && currTracerFeatures >= tracerFeaturesUpdateStep) {
                fd.update();
            }

            // Recoding update
            if (rd != null) {
                rd.update();
            }

            // Log the number of epochs
            epoch++;

            // Update the cumulated error
            cumulatedError += err;
        }

        // Complete the logging progress
        if (sample >= SAMPLES) {
            System.out.println("100%]");
        }

        scae.trainingDone();

        return cumulatedError;
    }

    /**
     * Train an auto encoder which requires a supervised training type.
     * For example, LDAAutoEncoder needs class labels for his initial training.
     * This method is designed to train a supervised AE with the GT on his filename.
     * Examples of dataset like this are: CIFAR, MNIST.
     *
     * @param scae    the scae to be trained
     * @param element the element of this tag node
     * @param fd      the feature display object (may be null!)
     * @param rd      the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainSupervisedFileNameBasedAutoEncoder(SCAE scae, Element element, FeatureDisplay fd, RecodingDisplay rd) throws IOException {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Number of samples already evaluated
        int sample = 0;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Counter of epochs (logging purpose only)
        int epoch = 0;

        // Epoch-wise error
        double err;
        int epochSize;

        // Cumulated training error
        double cumulatedError = 0;

        // Random numbers generator
        Random rand = XMLScript.getRandom();

        /***********************************************************************************************
         * PARSE ELEMENT FROM XML
         **********************************************************************************************/

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

        /***********************************************************************************************
         * TRAIN AE
         **********************************************************************************************/
        startTime = System.currentTimeMillis();

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Iterate until enough samples has been evaluated
        while (sample < SAMPLES) {

            // Epoch-wise error
            err = 0;
            epochSize = 0;

            // Log every ~10% the progress of training
            if (((sample - 1) * 10) / SAMPLES >= loggingProgress) {
                System.out.print(loggingProgress * 10 + "% ");
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
                loggingProgress = (sample * 10) / SAMPLES + 1;
            }

            // Iterate over all classes (data balancing)
            for (int c = 0; c < nbClasses; c++) {

                DataBlock db = data.get(c).get((int) (Math.random() * data.get(c).size()));

                int x = 0;
                int y = 0;

                if (db.getWidth() - scae.getInputPatchWidth() > 0) {
                    x = rand.nextInt(db.getWidth() - scae.getInputPatchWidth());
                    y = rand.nextInt(db.getHeight() - scae.getInputPatchHeight());
                }

                // Set input
                scae.setInput(db, x, y);

                // Train
                err += scae.trainSupervised(c);

                // Increase counters
                sample++;
                epochSize++;

                // Stop execution if MAXTIME reached
                if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                    // Complete the logging progress
                    System.out.println("]");
                    script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                    scae.trainingDone();
                    return cumulatedError;
                }

            }

            // Add the new epoch point to the plot
            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err / epochSize);
            }

            // Feature display update
            if (fd != null && currTracerFeatures >= tracerFeaturesUpdateStep) {
                fd.update();
            }

            // Recoding update
            if (rd != null) {
                rd.update();
            }

            // Log the number of epochs
            epoch++;

            // Update the cumulated error
            cumulatedError += err;
        }

        // Complete the logging progress
        if (sample >= SAMPLES) {
            System.out.println("100%]");
        }

        scae.trainingDone();

        // Free memory
        data = null;
        System.gc();

        return cumulatedError;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // DENOISING AUTO ENCODER
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Train a denoising auto encoder
     *
     * @param scae the scae to be trained
     * @param ds   the noisy data set to train that will be used to train the scae
     * @param fd   the feature display object (may be null!)
     * @param rd   the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainDenoisingAutoEncoder(SCAE scae, NoisyDataset ds, FeatureDisplay fd, RecodingDisplay rd) {

        // Time of start of the execution, necessary to stop after max time has reached
        long startTime = System.currentTimeMillis();

        // Error epoch wise
        double err;

        // Logging purpose only variable
        int loggingProgress = 1;

        // Cumulated training error
        double cumulatedError = 0;

        // Number of samples already evaluated
        int sample = 0;

        // Epoch counter
        int epoch = 0;

        // Init the index array as index: [1,2,3,4,5,6...,n] where n = ds.size()
        int[] index = new int[ds.size()];
        for (int i=0; i<index.length; i++) {
            index[i] = i;
        }

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Iterate until enough samples has been evaluated
        while (sample < SAMPLES) {
            //for (int epoch=0; epoch<SAMPLES; epoch+=index.length) {

            // Epoch-wise error
            err = 0;

            // Log every ~10% the progress of training
            if ((sample * 10) / SAMPLES >= loggingProgress) {
                System.out.print(loggingProgress * 10 + "% ");
                if (loggingProgress > 1) {
                    System.out.print(" ");
                }
                loggingProgress = (sample * 10) / SAMPLES + 1;
            }

            /* Shuffle the index array. The reason for this shuffle instead of the Dataset.shuffle() is that
             * we need the clean and noisy dataset to be shuffled in the same way otherwise we lose the
             * reference between clean and noisy data.
             */
            for (int i=0; i<index.length; i++) {
                int j = (int)(index.length*Math.random());
                int k = index[i];
                index[i] = index[j];
                index[j] = k;
            }

            // For each epoch
            for (int n : index) {
                DataBlock clean = ds.getClean(n);
                DataBlock noisy = ds.getNoisy(n);

                // Get random pixel
                int x = (int) (Math.random() * (clean.getWidth() - scae.getInputPatchWidth()));
                int y = (int) (Math.random() * (clean.getHeight() - scae.getInputPatchHeight()));

                // Train scae
                err += scae.trainDenoising(clean, noisy, x, y);

                // Increase counter of examined samples
                sample++;

                currTracerFeatures++;
            }

            // Add the new epoch point to the plot
            if (tracer != null) {
                // Log the error at each epoch
                tracer.addPoint(sample, err);
            }

            // Feature display update
            if (fd != null && currTracerFeatures >= tracerFeaturesUpdateStep) {
                fd.update();
            }

            // Recoding update
            if (rd!=null) {
                rd.update();
            }

            // Log the number of epochs
            epoch++;

            // Update the cumulated error
            cumulatedError += err;

            // Stop execution if MAXTIME reached
            if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                script.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                scae.trainingDone();
                return cumulatedError;
            }

        }

        // Complete the logging progress
        if (sample < SAMPLES) {
            System.out.println("]");
        } else {
            System.out.println("100%]");
        }
        scae.trainingDone();

        return cumulatedError;
    }

    @Override
    public String tagName() {
        return "train-scae";
    }
}
