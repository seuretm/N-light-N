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

import org.jtransforms.fft.FloatFFT_2D;

/**
 *
 * @author Mathias Seuret
 */
public class FFT2D extends SpectralTransform {
    FloatFFT_2D fft;
    
    /**
     * Constructs a channel-wise frequency transform.
     * @param w width of the data
     * @param h height of the data
     * @param d depth (number of channels) of the data
     */
    public FFT2D(int w, int h, int d) {
        super(w, h, d);
        fft = new FloatFFT_2D(w, h);
    }

    @Override
    protected void realForward(float[][][] a) {
        for (int z=0; z<depth; z++) {
            fft.realForward(a[z]);
        }
    }

    @Override
    protected void realInverse(float[][][] a) {
        for (int z=0; z<depth; z++) {
            fft.realInverse(a[z], true);
        }
    }
}
