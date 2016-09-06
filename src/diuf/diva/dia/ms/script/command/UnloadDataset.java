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
 * This command allows the user to un-load dataset which are no longer being used. An example would be a script in which
 * a classifier gets trained and then evaluated. The training dataset can be unloaded to free memory for the evaluating
 * phase.
 * <p>
 * XML syntax:
 * <p>
 * <unload-dataset id="stringID"/>
 *
 * @author Michele Alberti
 */
public class UnloadDataset extends AbstractCommand {

    public UnloadDataset(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String id = readAttribute(element, "id");
        script.println("Unloading dataset: " + id);
        script.datasets.remove(id);
        return "";
    }

    @Override
    public String tagName() {
        return "unload-dataset";
    }

}
