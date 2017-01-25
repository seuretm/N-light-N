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

package diuf.diva.dia.ms.util;

import diuf.diva.dia.ms.ml.ae.scae.SCAE;

/**
 * Displays the features of a SCAE.
 * @author Mathias Seuret
 */
public class FeatureDisplay extends DataBlockDisplay {
    SCAE scae;
    
    /**
     * Constructor of the class, takes as parameter the SCAE which features
     * will have to be displayed.
     * @param scae considered SCAE
     */
    public FeatureDisplay(SCAE scae) {
        super(scae.extractFeatures(), "Features");
        this.scae = scae;
    }
    
    @Override
    public void update() {
        DataBlock db = scae.extractFeatures();
        updateImage(db);
        repaint();
    }
}
