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

package diuf.diva.dia.ms.ml.ae.pooler;

import diuf.diva.dia.ms.util.DataBlock;
import java.io.Serializable;

/**
 *
 * @author Mathias Seuret
 */
public abstract class PoolerSelector implements Serializable {
    public final int inputWidth;
    public final int inputHeight;
    public PoolerSelector(int inputWidth, int inputHeight) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
    }
    
    public float unselect(float value) {
        return value;
    }
    
    public void learn() {
        // Nothing done by default
    }
    
    public abstract float select(DataBlock input, int channel, int posX, int posY);
    public abstract void backPropagate(float error, DataBlock prevErr, DataBlock input, int inputX, int inputY, int inputZ);
}
