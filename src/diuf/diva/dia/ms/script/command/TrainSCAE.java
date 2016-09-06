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
import org.jdom2.Element;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Random;

/**
 * Trains an autoencoder or a denoising autoencoder. It allows to display and save on file a plot of the training
 * error.
 *
 * XML syntax to use this feature:
 *
 *  <train-scae ref="myScae">
 *      <dataset>ds</dataset>
 *      <samples>SAMPLES</samples>
 *      <max-time>MAXTIME</max-time>
 *      <display-features>200</display-features> 			// optional
 *      <display-recoding>stringPATH</display-recoding> 	// optional
 *      <display-progress>200</display-progress> 			// optional
 *      <save-progress>stringPATH</save-progress> 			// optional: but make no sense if display progress is not there
 *  </train-scae>
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class TrainSCAE extends AbstractCommand {

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

        if (ds == null && nds == null) {
            error("cannot find dataset " + dataset);
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
                tracer = new Tracer("SCAE training error", "Epoch", "Error", expectedSamples, element.getChild("display-progress") != null);
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

        // Training an AE
        System.out.print(new SimpleDateFormat("HH:mm:ss.SSS").format(new Date()) + ": " +
                "SCAE Starting training {maximum time:" + MAXTIME + "m, samples:" + SAMPLES + "} Progress[");

        // Return value of the function
        String returnValue = null;

        // Query the normal datasets
        if (ds!=null) {
            // Verify dataset size before training
            for (DataBlock db : ds) {
                if (db.getWidth() < scae.getInputPatchWidth() || db.getHeight() < scae.getInputPatchHeight()) {
                    error(
                            "an image of the dataset is smaller than the input of the autoencoder:\n" +
                                    "[" + db.getWidth() + "x" + db.getHeight() + "]<[" + scae.getInputPatchWidth() + "x" + scae.getInputPatchHeight() + "]"
                    );
                }
            }
            returnValue = String.valueOf(trainAutoencoding(scae, ds, featureDisplay, recodingDisplay));
        }

        // Query the noisy dataset
        if (nds!=null) {
            // Verify dataset size before training (on clean only is sufficient!)
            for (DataBlock db : nds.clean) {
                if (db.getWidth() < scae.getInputPatchWidth() || db.getHeight() < scae.getInputPatchHeight()) {
                    error(
                            "an image of the dataset is smaller than the input of the autoencoder:\n" +
                                    "[" + db.getWidth() + "x" + db.getHeight() + "] < [" + scae.getInputPatchWidth() + "x" + scae.getInputPatchHeight() + "]"
                    );
                }
            }
            returnValue = String.valueOf(trainDenoising(scae, nds, featureDisplay, recodingDisplay));
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

    /**
     * Train an auto encoder
     *
     * @param scae the scae to be trained
     * @param ds   the data set to train that will be used to train the scae
     * @param fd   the feature display object (may be null!)
     * @param rd   the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainAutoencoding(SCAE scae, Dataset ds, FeatureDisplay fd, RecodingDisplay rd) {

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
        
        // Random numbers generator
        Random rand = new Random();

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

                int x = rand.nextInt(db.getWidth() - scae.getInputPatchWidth());
                int y = rand.nextInt(db.getHeight() - scae.getInputPatchHeight());

                // Set input
                scae.setInput(db, x, y);

                // Train
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
                break;
            }

        }

        // Complete the logging progress
        if (sample >= SAMPLES) {
            System.out.println("100%]");
        }

        scae.trainingDone();

        return cumulatedError;
    }

    /**
     * Train a denoising auto encoder
     *
     * @param scae the scae to be trained
     * @param ds   the noisy data set to train that will be used to train the scae
     * @param fd   the feature display object (may be null!)
     * @param rd   the recoding display object (may be null!)
     * @return cumulated error of the training
     */
    private double trainDenoising(SCAE scae, NoisyDataset ds, FeatureDisplay fd, RecodingDisplay rd) {

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
                break;
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
