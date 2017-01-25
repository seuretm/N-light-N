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

package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * Adds a layer to a SCAE. Described in the doc.
 * @author Mathias Seuret
 */
public class AddLayer extends CreateStackedAE {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public AddLayer(XMLScript script) {
        super(script);
    }
    
    /**
     * @param id of an SCAE
     * @return the SCAE or throws an error if not found
     */
    @Override
    protected SCAE getAE(String id) {
        SCAE ae = script.scae.get(id);
        if (ae==null) {
            error("cannot find an autoencoder with id "+id);
        }
        return ae;
    }
    
    /**
     * Returns the ID referenced by an element.
     * @param e element
     * @return an ID
     */
    @Override
    protected String readId(Element e) {
        return readAttribute(e, "ref");
    }
    
    @Override
    protected SCAE process(SCAE scae, AutoEncoder unit, int ox, int oy) {
        scae.addLayer(
                unit,
                ox,
                oy
        );
        return scae;
    }

    @Override
    public String tagName() {
        return "add-layer";
    }
}
