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

import java.io.FileInputStream;
import java.io.ObjectInputStream;

/**
 * Loads an SCAE or a classifier. Described in the documentation.
 * @author Mathias Seuret, Michele Alberti
 */
public class Load extends AbstractCommand {

    public Load(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String id    = readAttribute(element, "id");
        String fName = readElement(element, "file");

        script.println("Loading: " + readAttribute(element, "id"));

        Object o;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fName))) {
            o = ois.readObject();
        }
        
        if (o instanceof SCAE) {
            script.scae.put(id, (SCAE) o);
            return "";
        }

        if (o instanceof Classifier) {
            script.classifiers.put(id, (Classifier) o);
            return "";
        }
        
        error("Cannot recognize the class of the object in "+fName);
        
        return "";
    }

    @Override
    public String tagName() {
        return "load";
    }
    
}
