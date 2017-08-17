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

package diuf.diva.dia.ms.ml.ae.frequency;

import org.jtransforms.dst.FloatDST_2D;

/**
 * @author Mathias Seuret
 */
public class DST2D extends SpectralTransform {
    FloatDST_2D dst;
    
    public DST2D(int w, int h, int d) {
        super(w, h, d);
        dst = new FloatDST_2D(w, h);
    }

    @Override
    protected void realForward(float[][][] a) {
        for (int z=0; z<depth; z++) {
            dst.forward(a[z], true);
        }
    }

    @Override
    protected void realInverse(float[][][] a) {
        for (int z=0; z<depth; z++) {
            dst.inverse(a[z], true);
        }
    }
    
}
