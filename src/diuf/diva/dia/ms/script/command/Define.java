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

import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * Defines a variable. Described in the doc.
 * @author Mathias Seuret, Michele Alberti
 */
public class Define extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public Define(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String key = element.getAttributeValue("id");
        if (key==null) {
            error("an id attribute is required");
        }

        String val = element.getText();
        if (val==null || val.length()==0) {
            if (!script.definitions.containsKey(key)) {
                error(key + " is empty. Therefore it should be defined by command line");
            }
            return "";
        }
        script.definitions.put(key, val);
        return "";
    }

    @Override
    public String tagName() {
        return "define";
    }
    
}
