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

package diuf.diva.dia.ms.ml.ae;

import diuf.diva.dia.ms.ml.ae.frequency.SpectralTransform;
import diuf.diva.dia.ms.ml.layer.Layer;
import diuf.diva.dia.ms.util.DataBlock;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;

/**
 * This is an AutoEncoder which allows to apply spectral transforms to its
 * input before encoding, and inverse transforms after decoding. Note that
 * different transforms can be used for the forward and inverse directions.
 * @author Mathias Seuret
 */
public class SpectralAutoEncoder2 extends AutoEncoder implements Serializable {

    protected SpectralTransform forwardTransform;
    protected SpectralTransform inverseTransform;
    protected float[] freqDecoded;
    protected float[] spatialInput;
    protected float[] freqTarget;
    protected boolean isTraining = false;
    
    public SpectralAutoEncoder2(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int outputDepth,
            String forwardTransform,
            String inverseTransform,
            String encoder,
            String decoder
    ) {
        super(inputWidth, inputHeight, inputDepth, outputDepth);
        
        freqDecoded  = new float[inputLength];
        spatialInput = new float[inputLength];
        freqTarget   = new float[inputLength];
        

        try {
            Class fc = Class.forName("diuf.diva.dia.ms.ml.ae.frequency." + forwardTransform);
            Class ic = Class.forName("diuf.diva.dia.ms.ml.ae.frequency." + inverseTransform);
            this.forwardTransform = (SpectralTransform)fc.getDeclaredConstructor(
                    int.class,
                    int.class,
                    int.class
            ).newInstance(inputWidth, inputHeight, inputDepth);
            this.inverseTransform = (SpectralTransform)ic.getDeclaredConstructor(
                    int.class,
                    int.class,
                    int.class
            ).newInstance(inputWidth, inputHeight, inputDepth);
            
            
            Class ec = Class.forName("diuf.diva.dia.ms.ml.layer." + encoder);
            Class dc = Class.forName("diuf.diva.dia.ms.ml.layer." + decoder);
            
            // Setting the encoder
            setEncoder(
                    (Layer) ec.getDeclaredConstructor(
                            float[].class,
                            int.class,
                            int.class,
                            float[][].class,
                            float[].class
                    ).newInstance(
                            null,
                            inputWidth * inputHeight * inputDepth,
                            outputDepth,
                            null,
                            null
                    )
            );
            // Setting the decoder
            setDecoder(
                    (Layer) dc.getDeclaredConstructor(
                            float[].class,
                            int.class,
                            int.class,
                            float[][].class,
                            float[].class
                    ).newInstance(
                        null,
                        outputDepth,
                        inputWidth * inputHeight * inputDepth,
                        null,
                        null
                    )
            );
        } catch (
                SecurityException
                        | InstantiationException
                        | IllegalAccessException
                        | IllegalArgumentException
                        | InvocationTargetException
                        | ClassNotFoundException
                        | NoSuchMethodException ex
                ) {
            throw new Error(ex.getMessage());
        }
    }
    
    @Override
    public void setDecoder(Layer l) {
        super.setDecoder(l);
        l.setOutputArray(freqDecoded);
    }
    
    /**
     * Gets the input patch, transform it and put it into the array
     * @param array destination array
     */
    @Override
    protected void inputPatchToArray(float[] array) {
        super.inputPatchToArray(spatialInput);
        forwardTransform.forward(spatialInput, array);
    }
    
    public DataBlock forwardTransform(DataBlock src, int ox, int oy) {
        DataBlock db = new DataBlock(inputWidth, inputHeight, inputDepth);
        forwardTransform.forward(src, ox, oy, db, 0, 0);
        return db;
    }
    
    public DataBlock inverseTransform(DataBlock src, int ox, int oy) {
        DataBlock db = new DataBlock(inputWidth, inputHeight, inputDepth);
        inverseTransform.inverse(src, ox, oy, db, 0, 0);
        return db;
    }
    
    @Override
    public float train() {
        assert (decoder.getPreviousError() != null);

        // Compute output
        encode();

        if (forwardTransform.getClass().equals(inverseTransform.getClass())) {
            // If same forward/inverse, no need to inverse decoded values
            decoder.compute();
            for (int i = 0; i < inputLength; i++) {
                decoder.setExpected(i, inputArray[i]);
            }
        } else {
            decode();
            inverseTransform.inverse(spatialInput, freqTarget);
            for (int i = 0; i < inputLength; i++) {
                decoder.setExpected(i, freqTarget[i]);
            }
        }
        
        // Backpropagate
        float err = decoder.backPropagate();
        encoder.backPropagate();
        
        encoder.clearError();
        decoder.clearError();

        // Learn
        encoder.learn();
        decoder.learn();

        return err;
    }
    
    /**
     * Backpropagate the error, if needed, through frequency modification
     * @return the mean absolute error of the top layer
     */
    @Override
    public float backPropagate() {
        // Backpropagate
        float err = encoder.backPropagate();

        // Set the previous error from layer to datablock!
        if (prevErr!=null) {
            float[] e = encoder.getPreviousError();
            forwardTransform.inverse(e, e);
            prevErr.weightedPatchPaste(e, inputX, inputY, inputWidth, inputHeight);
        }
        return err;
    }
    
    public void decode() {
        super.decode();
        inverseTransform.inverse(freqDecoded, decoded);
    }

    @Override
    public String getTypeName() {
        return "[FAE]";
    }

    @Override
    public AutoEncoder clone() {
        // Create a StandardAutoEncoder the standard way
        SpectralAutoEncoder2 autoEncodeer = new SpectralAutoEncoder2(
                inputWidth,
                inputHeight,
                inputDepth,
                outputDepth,
                parseClassName(forwardTransform.getClass()),
                parseClassName(inverseTransform.getClass()),
                parseClassName(encoder.getClass()),
                parseClassName(decoder.getClass())
        );

        // Set input
        autoEncodeer.setInput(input, inputX, inputY);

        // Set output
        autoEncodeer.setOutput(output, 0, 0);

        /* If error has not the same size of output, it means it was not used before.
         * Hence we do not set it. It is clear that one has to set the error manually after
         */
        if (output.getWidth() == error.getWidth() && output.getHeight() == error.getHeight() && output.getDepth()==error.getDepth()) {
            autoEncodeer.setError(error);
        }

        // Set previous error if not null
        if (prevErr != null) {
            autoEncodeer.setPrevError(prevErr);
        }

        // Set the encoder / decoder
        autoEncodeer.setEncoder(encoder.clone());
        autoEncodeer.setDecoder(decoder.clone());

        return autoEncodeer;
    }

    @Override
    public void startTraining() {
        encoder.startTraining();
        decoder.startTraining();
        isTraining = true;
    }

    @Override
    public void stopTraining() {
        encoder.stopTraining();
        decoder.stopTraining();
        isTraining = false;
    }

    @Override
    public boolean isTraining() {
        return isTraining;
    }

    @Override
    public void clearGradient() {
        encoder.clearGradient();
        decoder.clearGradient();
    }
}
