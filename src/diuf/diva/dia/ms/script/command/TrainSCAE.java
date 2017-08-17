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
import diuf.diva.dia.ms.util.misc.Pixel;
import org.jdom2.Element;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;

/**
 * Trains an autoencoder or a denoising autoencoder. It allows to display and save on file a plot of the training
 * error.
 *
 * XML syntax to use this feature:
 *
 *  &lt;train-scae ref="myScae"&gt;
 *      &lt;dataset&gt;ds&lt;/dataset&gt;
 *      &lt;groundTruth&gt;gt&lt;/groundTruth&gt;                       // optional: only for supervised AE
 *      &lt;samples&gt;SAMPLES&lt;/samples&gt;
 *      &lt;max-time&gt;MAXTIME&lt;/max-time&gt;
 *      &lt;!-- optional --&gt;
 *      &lt;display-features&gt;200&lt;/display-features&gt;
 *      &lt;!-- optional --&gt;
 *      &lt;display-recoding&gt;stringPATH&lt;/display-recoding&gt;
 *      &lt;!-- optional --&gt;
 *      &lt;display-progress&gt;200&lt;/display-progress&gt;
 *      &lt;!-- optional, but needs display-progress --&gt;
 *      &lt;save-progress&gt;stringPATH&lt;/save-progress&gt;
 *  &lt;/train-scae&gt;
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class TrainSCAE extends AbstractCommand {

    /**
     * Support variables for the &lt;display-features&gt; tag
     */
    int tracerFeaturesUpdateStep = 1000;
    int currTracerFeatures = 0;
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
     * Constructor of the class.
     * @param script which creates the command
     */
    public TrainSCAE(XMLScript script) {
        super(script);
    }

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
        HashMap<Integer, ArrayList<DataBlock>> fnbds = script.fileNameBasedDatasets.get(dataset);

        if (ds == null && nds == null && fnbds == null) {
            error("cannot find dataset " + dataset);
        }

        // Parse parameter for training
        this.SAMPLES = Integer.parseInt(readElement(element, "samples"));
        this.MAXTIME = (element.getChild("max-time") != null) ? Integer.parseInt(readElement(element, "max-time")) : 0;

        if (SAMPLES == 0 && MAXTIME == 0) error("cannot set both epochs and max-time to 0");
        if (SAMPLES == 0) MAXTIME = Integer.MAX_VALUE;
        if (MAXTIME == 0) MAXTIME = Integer.MAX_VALUE;

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
//                tracer = new Tracer(
//                        "SCAE training error",
//                        "Epoch",
//                        "Error",
//                        expectedSamples,
//                        element.getChild("display-progress") != null
//                );
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
            for (int i = 0; i < ds.size(); i++) {
                DataBlock db = ds.get(i);
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

            if (scae.isSupervised()) {
                returnValue = String.valueOf(trainSupervisedAutoEncoder(scae, ds, featureDisplay, recodingDisplay));
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

        // Query the file name based dataset
        if (fnbds != null) {
            returnValue = String.valueOf(trainSupervisedFileNameBasedAutoEncoder(scae, fnbds, featureDisplay, recodingDisplay));
        }

        // If a result has been computed
        if (returnValue != null) {
            // If tracer has been used
            if (tracer != null) {
//                tracer.addRawData();
//                tracer.addCumulatedAverage();
//                tracer.addMovingAverage();
//                tracer.addMovingMedian();
                // Show the plot
                //tracer.display();

                // If requires, save the plot on disk at the provided path
                if (element.getChild("save-progress") != null) {
                    try {
                        //         tracer.savePlot(readElement(element, "save-progress"));
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
    private double trainAutoencoder(SCAE scae, Dataset<DataBlock> ds, FeatureDisplay fd, RecodingDisplay rd) {

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

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Iterate until enough samples has been evaluated
        scae.startTraining();
        while (sample <= SAMPLES) {

            // Shuffle the dataset at each epoch
            ds.shuffle();

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
                    x = Random.nextInt(db.getWidth() - scae.getInputPatchWidth());
                }
                if (db.getHeight()>scae.getInputPatchHeight()) {
                    y = Random.nextInt(db.getHeight() - scae.getInputPatchHeight());
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
                XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
                scae.trainingDone();
                return cumulatedError;
            }

        }

        scae.trainingDone();

        scae.stopTraining();

        // Complete the logging progress
        if (sample >= SAMPLES) {
            System.out.println("100%]");
        }

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
     * @param ds the dataset to train that will be used to train the scae
     * @param fd the feature display object (may be null!)
     * @param rd the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainSupervisedAutoEncoder(SCAE scae, Dataset<GroundTruthDataBlock> ds, FeatureDisplay fd, RecodingDisplay rd) {

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
            for (GroundTruthDataBlock db : ds) {
                // For each image, take a sample out of each class -> data balancing
                for (int c = 0; c <= db.getNumberOfClasses(); c++) {

                    // Get next representative
                    Pixel p = db.getRandomRepresentative(c);

                    // If pixel 'p' is null it means that this specific GT does not contain this class
                    if (p == null) {
                        continue;
                    }

                    // If its a boundary convert it to background
                    if (((db.getGt(p.x, p.y) >> 23) & 0x1) == 1) {
                        c = 0x1;
                    }

                    // Set input
                    scae.centerInput(db, p.x, p.y);

                    // Train
                    err += scae.trainSupervised(c);

                    // Increase counters
                    sample++;
                    epochSize++;

                    // Stop execution if MAXTIME reached
                    if (((int) (System.currentTimeMillis() - startTime) / 60000) >= MAXTIME) {
                        // Complete the logging progress
                        System.out.println("]");
                        XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
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
     * @param fd      the feature display object (may be null!)
     * @param rd      the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainSupervisedFileNameBasedAutoEncoder(SCAE scae, HashMap<Integer, ArrayList<DataBlock>> fnbds, FeatureDisplay fd, RecodingDisplay rd) throws IOException {

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

        /**
         * Final number of classes found in the picture
         */
        final int nbClasses = fnbds.size();

        /***********************************************************************************************
         * TRAIN AE
         **********************************************************************************************/
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

                DataBlock db = fnbds.get(c).get((int) (Math.random() * fnbds.get(c).size()));

                int x = 0;
                int y = 0;

                if (db.getWidth() - scae.getInputPatchWidth() > 0) {
                    x = Random.nextInt(db.getWidth() - scae.getInputPatchWidth());
                    y = Random.nextInt(db.getHeight() - scae.getInputPatchHeight());
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
                    XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
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
                XMLScript.println("Maximum training time (" + MAXTIME + ") reached after " + epoch + " epochs");
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
