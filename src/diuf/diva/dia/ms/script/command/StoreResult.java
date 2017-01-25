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
 * @author Mathias Seuret
 */
public class StoreResult extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public StoreResult(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String ref = element.getAttributeValue("id");
        if (ref==null) {
            error(
                    "id attribute required"
            );
        }
        script.definitions.put(ref, script.definitions.get("$ANS"));
        return "";
    }

    @Override
    public String tagName() {
        return "store-result";
    }
    
}
