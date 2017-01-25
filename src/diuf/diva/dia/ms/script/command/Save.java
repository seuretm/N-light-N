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

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * Saves an autoencoder or a classifier.
 * @author Mathias Seuret, Michele Alberti
 */
public class Save extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public Save(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String ref   = readAttribute(element, "ref");
        String fName = readElement(element, "file");
        
        SCAE scae = script.scae.get(ref);
        if (scae!=null) {
            scae.save(fName);
            return "";
        }

        Classifier classifier = script.classifiers.get(ref);
        if (classifier != null) {
            classifier.save(fName);
            return "";
        }

        error("Cannot find "+ref);

        return "";
    }

    @Override
    public String tagName() {
        return "save";
    }
    
}
