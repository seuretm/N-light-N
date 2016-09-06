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

import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.DataBlock;
import org.jdom2.Element;

import java.io.File;

/**
 * Saves into a file the learned features. Described in the doc.
 * @author Mathias Seuret, Alberti Michele
 */
public class ShowFeatures extends AbstractCommand {

    public ShowFeatures(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {

        // Getting the SCAE
        String ref = readAttribute(element, "ref");
        SCAE ae = script.scae.get(ref);

        // Parsing file name
        String fName = readElement(element, "file");

        // Check whether the path is existing, if not create it
        File file = new File(fName);
        if (!file.isDirectory()) {
            file = file.getParentFile();
        }
        if (file!=null && !file.exists()) {
            file.mkdirs();
        }

        // Parse scale
        int scale  = Integer.parseInt(readElement(element, "scale"));
        if (scale<1) {
            error("the scale must be greater than 0");
        }

        script.println("SCAE Showing features " + fName);

        DataBlock db = ae.extractFeatures();
        db.setColorspace(script.colorspace);
        db.getImage().getScaled(scale).write(fName);

        return "";
    }

    @Override
    public String tagName() {
        return "show-features";
    }
    
}
