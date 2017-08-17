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

/**
 *
 * @author Mathias Seuret
 */
public class Max extends PoolerSelector {

    public Max(int inputWidth, int inputHeight) {
        super(inputWidth, inputHeight);
    }

    @Override
    public float select(DataBlock input, int channel, int inputX, int inputY) {
        float max = Float.NEGATIVE_INFINITY;
        for (int ox=0; ox<inputWidth; ox++) {
            for (int oy=0; oy<inputHeight; oy++) {
                max = Math.max(max, input.getValue(channel, inputX+ox, inputY+oy));
            }
        }
        return max;
    }
    
    @Override
    public void backPropagate(float error, DataBlock prevErr, DataBlock input, int inputX, int inputY, int inputZ) {
        int eX = 0;
        int eY = 0;
        float ext = 0.0f;
        for (int ox=0; ox<inputWidth; ox++) {
            for (int oy=0; oy<inputHeight; oy++) {
                float e = input.getValue(inputZ, inputX+ox, inputY+oy);
                if (e>ext) {
                    ext = e;
                    eX = ox;
                    eY = oy;
                }
            }
        }
        prevErr.addValue(inputZ, inputX+eX, inputY+eY, error);
    }
}
