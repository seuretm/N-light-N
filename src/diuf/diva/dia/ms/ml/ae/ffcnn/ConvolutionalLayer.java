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
package diuf.diva.dia.ms.ml.ae.ffcnn;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.scae.Convolution;
import diuf.diva.dia.ms.util.DataBlock;


/**
 *
 * @author ms
 */
public abstract class ConvolutionalLayer {
    public abstract void setInput(DataBlock db, int posX, int posY);
    
    public abstract void compute();
    
    public abstract void setExpected(int z, float ex);
    
    public abstract void setExpected(int x, int y, int z, float ex);
    
    public abstract void addError(int x, int y, int z, float e);
    
    public abstract void learn();
    
    public abstract float backPropagate();
    
    public abstract AutoEncoder getAutoEncoder(int x, int y);
    
    public abstract void clearError();
    
    public abstract void setPrevError(DataBlock db);
    
    public abstract float getLearningSpeed();
    
    public abstract void setLearningSpeed(float s);
    
    public abstract DataBlock getOutput();

    public abstract DataBlock getError();

    public abstract int getInputWidth();

    public abstract int getInputHeight();

    public abstract DataBlock getPrevError();
}
