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
import diuf.diva.dia.ms.util.DataBlock;


/**
 *
 * @author ms
 */
public interface ConvolutionalLayer {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void setInput(DataBlock db, int posX, int posY);

    int getInputWidth();

    int getInputHeight();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Error
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void setPrevError(DataBlock db);

    DataBlock getPrevError();

    void addError(int x, int y, int z, float e);

    void clearError();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void setExpected(int z, float ex);

    void setExpected(int x, int y, int z, float ex);

    DataBlock getError();

    void learn();

    float backPropagate();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Compute & output
    ///////////////////////////////////////////////////////////////////////////////////////////////
    void compute();

    DataBlock getOutput();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    AutoEncoder getAutoEncoder(int x, int y);

    int getXoffset();

    int getYoffset();

    float getLearningSpeed();

    void setLearningSpeed(float s);



}
