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

package diuf.diva.dia.ms.ml.rbm;

import java.io.Serializable;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.random;
import static java.lang.Math.signum;
import static java.lang.Math.sqrt;

/**
 * Basig Gaussian-Binary RBM, based on "Improved Learning of
 * Gaussian-Bernoulli Restricted Boltzmann Machines", Cho et
 * al, 2011.
 * @author Mathias Seuret
 */
public class BasicGBRBM implements Serializable {
    /**
     * Number of visible units.
     */
    int nbVisible;
    
    /**
     * Number of hidden units.
     */
    int nbHidden;
    
    /**
     * Stores the values of the visible units.
     */
    float[] visible;
    
    /**
     * Stores the values of the hidden units.
     */
    int[] hidden;
    
    /**
     * Weights.
     */
    float[][] w;
    
    /**
     * Bias for the visible units.
     */
    float[] b;
    
    /**
     * Bias for the hidden units.
     */
    float[] c;
    
    /**
     * Log of the variance of the visible units.
     */
    float[] z;
    
    /**
     * Variance of the visible units during a computation. It
     * is stored in an array in order to avoid to call Math.exp
     * a great number of times.
     */
    float[] eZ;
    
    /**
     * Positive values for the weights.
     */
    float[][] pW;
    
    /**
     * Negative values for the weights.
     */
    float[][] nW;
    
    /**
     * Positive values for the bias of the visible units.
     */
    float[] pB;
    
    /**
     * Negative values for the bias of the visible units.
     */
    float[] nB;
    
    /**
     * Positive values for the bias of the hidden units.
     */
    float[] pC;
    
    /**
     * Negative values for the bias of the hidden units.
     */
    float[] nC;
    
    /**
     * Positive values for the log of the variance of the visible units.
     */
    float[] pZ;
    
    /**
     * Negative values for the log of the variance of the visible units.
     */
    float[] nZ;
    
    /**
     * Learning speed
     */
    float eps = 1e-4f;
    
    /**
     * Constructs a GBRBM.
     * @param nbVisible number of visible units
     * @param nbHidden number of hidden units
     */
    public BasicGBRBM(int nbVisible, int nbHidden) {
        assert (nbVisible>=1);
        assert (nbHidden>=1);
        
        this.nbVisible = nbVisible;
        this.nbHidden  = nbHidden;
        visible = new float[nbVisible];
        hidden  = new int[nbHidden];
        w       = new float[nbVisible][nbHidden];
        b       = new float[nbVisible];
        c       = new float[nbHidden];
        z       = new float[nbVisible];
        eZ      = new float[nbVisible];
        
        pW      = new float[nbVisible][nbHidden];
        pB      = new float[nbVisible];
        pC      = new float[nbHidden];
        pZ      = new float[nbVisible];
        
        nW      = new float[nbVisible][nbHidden];
        nB      = new float[nbVisible];
        nC      = new float[nbHidden];
        nZ      = new float[nbVisible];
        
        // init weights
        for (int v=0; v<nbVisible; v++) {
            for (int h=0; h<nbHidden; h++) {
                w[v][h] = randomInitialWeight();
            }
        }
        
        for (int v=0; v<nbVisible; v++) {
            z[v] = 1;
            b[v] = randomInitialWeight();
        }
    }
    
    /**
     * Generates a random weight between -inf and +inf, with most of
     * the values between -0.04 and +0.04.
     * @return a float
     */
    private float randomInitialWeight() {
        return (float)(sqrt(-2*0.0001*log(random()))*signum(random()-0.5));
    }
    
    /**
     * @return the visible values
     */
    public float[] getVisible() {
        return visible;
    }
    
    /**
     * @return the hidden values
     */
    public int[] getHidden() {
        return hidden;
    }
    
    /**
     * Loads a sample.
     * @param sample array containing nbVisible values
     */
    public void load(float[] sample) {
        assert (sample.length==nbVisible);
        
        for (int i=0; i<nbVisible; i++) {
            visible[i] = sample[i];
        }
    }
    
    /**
     * Trains the RBM on a given sample.
     * @param sample a float array
     * @return an estimation of the reconstruction error
     */
    public float train(float[] sample) {
        load(sample);
        
        updateHidden();
        computePos();
        
        float d = updateVisible();
        updateHidden();
        
        for (int n=0; n<2; n++) {
            updateVisible();
            updateHidden();
        }
        
        computeNeg();
        
        applyCD();
        
        return d;
    }
    
    /**
     * Computes the positive values.
     */
    public void computePos() {
        computePosNeg(pW, pB, pC, pZ);
    }
    
    /**
     * Computes the negative values.
     */
    public void computeNeg() {
        computePosNeg(nW, nB, nC, nZ);
    }
    
    /**
     * Generic method for computing positive and negative values.
     * @param pnW positive or negative weight array
     * @param pnB positive or negative bias of the visible units
     * @param pnC positive or negative bias of the hidden units
     * @param pnZ positive or negative variances of the visible units
     */
    public void computePosNeg(float[][] pnW, float[] pnB, float[] pnC, float[] pnZ) {
        for (int v=0; v<nbVisible; v++) {
            float s = (float)exp(z[v]);
            pnZ[v] = 0.5f*(visible[v]-b[v])*(visible[v]-b[v]);
            for (int h=0; h<nbHidden; h++) {
                pnW[v][h] = 1.0f/s*visible[v]*hidden[h];
                pnZ[v] -= visible[v]*hidden[h]*w[v][h];
            }
            pnB[v] = 1/s*visible[v];
        }
        for (int h=0; h<nbHidden; h++) {
            pnC[h] = hidden[h];
        }
    }
    
    /**
     * Applies the contrastive divergence.
     */
    public void applyCD() {
        for (int v=0; v<nbVisible; v++) {
            for (int h=0; h<nbHidden; h++) {
                w[v][h] += eps * (pW[v][h]-nW[v][h]);
            }
            b[v] += eps * (pB[v]-nB[v]);
            z[v] += eps * exp(-z[v])*(pZ[v]-nZ[v]);
        }
        for (int h=0; h<nbHidden; h++) {
            c[h] += eps * (pC[h]-nC[h]);
        }
    }
    
    /**
     * Updates the hidden units.
     */
    public void updateHidden() {
        for (int v=0; v<nbVisible; v++) {
            eZ[v] = (float)exp(z[v]);
        }
        for (int h=0; h<nbHidden; h++) {
            float sum = c[h];
            for (int v=0; v<nbVisible; v++) {
                sum += visible[v]*w[v][h]/eZ[v];
            }
            float p = sigmoid(sum);
            hidden[h] = (random()<p) ? 1 : 0;
        }
    }
    
    /**
     * Sigmoid function.
     * @param x value
     * @return the sigmoid of x
     */
    public float sigmoid(float x) {
        return 1.0f / (1.0f+(float)exp(-x));
    }
    
    /**
     * Updates the visible units.
     * @return the difference between new and old values
     */
    public float updateVisible() {
        float diff = 0;
        for (int v=0; v<nbVisible; v++) {
            float sum = b[v];
            for (int h=0; h<nbHidden; h++) {
                sum += hidden[h]*w[v][h];
            }
            float sign = (random()<0.5) ? -1.0f : 1.0f;
            float s = (float)exp(z[v]);
            float vis = b[v] + sum + sign*(float)sqrt(-2*s*log(random()));
            diff += abs(vis-visible[v]);
            visible[v] = vis;
        }
        return diff/nbVisible;
    }
    
    /**
     * Computes a visible vector from the hidden units, ignoring the variance
     * of the visible units.
     */
    public void decode() {
        for (int v=0; v<nbVisible; v++) {
            float sum = b[v];
            for (int h=0; h<nbHidden; h++) {
                sum += hidden[h]*w[v][h];
            }
            visible[v] = b[v] + sum;
        }
    }
    
    /**
     * Extracts the feature corresponding the the n-th hidden unit.
     * @param n hidden unit number
     * @return a float array
     */
    public float[] extractFeature(int n) {
        assert (n>0);
        assert (n<nbHidden);
        
        for (int h=0; h<nbHidden; h++) {
            hidden[h] = (n==h) ? 1 : 0;
        }
        decode();
        return getVisible().clone();
    }
    
    /**
     * @return a string representation of the RBM
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        
        sb.append(String.format("        Bias"));
        for (int h=0; h<nbHidden; h++) {
            sb.append(String.format("      H-%d", h));
        }
        sb.append(String.format("\nBias            "));
        for (int h=0; h<nbHidden; h++) {
            sb.append(String.format(" %8.5f", c[h]));
        }
        sb.append("\n");
        
        for (int v=0; v<nbVisible; v++) {
            sb.append(String.format("V-%d     %8.5f", v, b[v]));
            for (int h=0; h<nbHidden; h++) {
                sb.append(String.format(" %8.5f", w[v][h]));
            }
            sb.append("\n");
        }
        
        return sb.toString();
    }
}
