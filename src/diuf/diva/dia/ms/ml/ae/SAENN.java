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

import diuf.diva.dia.ms.ml.layer.NeuralLayer;

import java.io.Serializable;
import java.util.Arrays;

/**
 * This is the denoising autoencoder. 
 * @author Mathias Seuret
 */
public class SAENN implements Serializable {
    /**
     * Dimension of the output.
     */
    protected final int outputLength;
    
    /**
     * Number of values in the input
     */
    protected final int inputLength;
    
    /**
     * Input array
     */
    protected float[] input;
    
    /**
     * Encoding layer
     */
    private final NeuralLayer encoder;
    
    /**
     * Decoding layer
     */
    private final NeuralLayer decoder;
    
    /**
     * Stores the result - this is also the output array.
    */
    protected float[] encoded;
    
    /**
     * Stores the decoded result.
     */
    protected float[] decoded;
    
    /**
     * Constructor
     * @param inputLength size of the input array
     * @param outputLength size of the output array
     */
    public SAENN(int inputLength, int outputLength) {
        this(inputLength, outputLength, null, null, null, null);
    }
    
    /**
     * Constructor, specifying weights and bias.
     * @param inputLength number of inputs
     * @param outputLength number of outputs
     * @param eWeight encoding weights
     * @param dWeight decoding weights
     * @param eBias encoding bias
     * @param dBias decoding bias
     */
    public SAENN(int inputLength, int outputLength, float[][] eWeight, float[][] dWeight, float[] eBias, float[] dBias) {
        this.inputLength  = inputLength;
        this.outputLength = outputLength;
        this.input        = new float[inputLength];

        encoder = new NeuralLayer(null, inputLength, outputLength, eWeight, eBias);
        decoder = new NeuralLayer(null, outputLength, inputLength, dWeight, dBias);
        
        encoded = new float[outputLength];
        decoded = new float[inputLength];
        
        encoder.setInputArray(input);
        encoder.setOutputArray(encoded);
        
        decoder.setInputArray(encoded);
        decoder.setOutputArray(decoded);
        
        encoder.setPreviousError(null);
        decoder.setPreviousError(encoder.getError());
    }
    
    /**
     * Trains the unit.
     * @return  the mean error
     */
    public float train() {
        // Compute output
        getEncoder().compute();
        getDecoder().compute();
        
        for (int i=0; i<inputLength; i++) {
            getDecoder().setExpected(i, input[i]);
        }

        float err = getDecoder().backPropagate();
        getEncoder().learn();
        getDecoder().learn();
        return err;
    }
    
    /**
     * Trains the decoder only.
     * @return the decoding error
     */
    public float trainDecoder() {
        getEncoder().compute();
        getDecoder().compute();
        
        for (int i=0; i<inputLength; i++) {
            getDecoder().setExpected(i, input[i]);
        }

        getDecoder().learn();
        return -1;
    }

    /**
     * Encodes the input, stores the result in the encoded array.
     */
    public void encode() {
        getEncoder().compute();
    }

    /**
     * Decodes the encoded array, stores the result in input.
     */
    public void decode() {
        getDecoder().compute();
    }
    
    /**
     * @return the array of encoded values
     */
    public float[] getEncoded() {
        return encoded;
    }
    
    /**
     * Sets the array used as input - it's not copied.
     * @param in the array to use
     */
    public void setInput(float[] in) {
        assert (in.length==inputLength);
        
        input = in;
        getEncoder().setInputArray(in);
    }

    /**
     * @return the decoded data
     */
    public float[] getDecoded() {
        return decoded;
    }

    /**
     * @return the encoder
     */
    public NeuralLayer getEncoder() {
        return encoder;
    }

    /**
     * @return the decoder
     */
    public NeuralLayer getDecoder() {
        return decoder;
    }

    /**
     * Delete some of the features of the autoencoder.
     * @param number either a list of feature numbers, or an array
     */
    public void deleteFeatures(int... number) {
        int[] num = number.clone();
        Arrays.sort(num);
        for (int i=num.length-1; i>=0; i--) {
            encoder.deleteOutput(num[i]);
            decoder.deleteInput(num[i]);
        }
    }
    
}
