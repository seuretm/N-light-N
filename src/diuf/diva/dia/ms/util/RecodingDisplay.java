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
import diuf.diva.dia.ms.script.command.Recode;

/**
 * Shows the recoded version of an image.
 * @author Mathias Seuret
 */
public class RecodingDisplay extends DataBlockDisplay {
    DataBlock input;
    SCAE scae;
    
    /**
     * Constructs an instance.
     * @param db datablock for the reconstruction.
     * @param scae the autoencoder
     */
    public RecodingDisplay(DataBlock db, SCAE scae) {
        super(db, "Sample reconstruction");
        input = db;
        this.db = new DataBlock(db.getWidth(), db.getHeight(), db.getDepth());
        this.scae = scae;
    }

    @Override
    public void update() {
        db = Recode.recode(scae, input, input.getColorspace());
        updateImage(db);
    }
    
}
