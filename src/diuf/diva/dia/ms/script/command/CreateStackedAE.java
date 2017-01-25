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

import diuf.diva.dia.ms.ml.ae.*;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * Creates an SCAE. Described in the doc.
 * @author Mathias Seuret, Michele Alberti
 */
public class CreateStackedAE extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public CreateStackedAE(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String id = readId(element);
        
        Element unitEl = element.getChild("unit");
        if (unitEl==null) {
            error("require unit tag");
        }
        if (unitEl.getChildren().size()!=1) {
            error("unit requires one child tag");
        }
        unitEl = unitEl.getChildren().get(0);
        
        String type = unitEl.getName();
        if (type==null) {
            error("unit requires a type");
        }
        
        SCAE scae = getAE(id);
        int inputDepth = (scae==null) ? script.colorspace.depth : scae.getOutputDepth();

        AutoEncoder unit = null;
        int width  = Integer.parseInt(readElement(element, "width"));
        int height = Integer.parseInt(readElement(element, "height"));
        int ox     = Integer.parseInt(readElement(element, "offset-x"));
        int oy     = Integer.parseInt(readElement(element, "offset-y"));

        if (type.equalsIgnoreCase("STANDARD")) {
            // Parse the 'dimensions' text
            int hidden = Integer.parseInt(readElement(unitEl, "hidden"));

            // Parse the 'layer' text
            
            String encoderClassName = null;
            String decoderClassName = null;
            if (unitEl.getChild("layer")!=null) {
                encoderClassName = readElement(unitEl, "layer");
                decoderClassName = encoderClassName;
            } else {
                encoderClassName = readElement(unitEl, "encoder");
                decoderClassName = readElement(unitEl, "decoder");
            }

            // Create the unit with specified parameters
            unit = new StandardAutoEncoder(
                    width,
                    height,
                    inputDepth,
                    hidden,
                    encoderClassName,
                    decoderClassName
            );
        }
        
        if (type.equalsIgnoreCase("MAX-POOLER")) {
            unit = new MaxPooler(
                    width,
                    height,
                    inputDepth
            );
        }
        
        if (type.equalsIgnoreCase("BasicBBRBM")) {
            int hidden = Integer.parseInt(readElement(unitEl, "hidden"));
            unit = new BBRBMUnit(
                    width,
                    height,
                    inputDepth,
                    hidden
            );
        }
        
        if (type.equalsIgnoreCase("BasicGBRBM")) {
            int hidden = Integer.parseInt(readElement(unitEl, "hidden"));
            unit = new GBRBMUnit(
                    width,
                    height,
                    inputDepth,
                    hidden
            );
        }
        
        if (type.equalsIgnoreCase("PCA")) {
            // Parse the 'dimensions' text
            String s = readElement(unitEl, "dimensions");
            int dim;
            if (s.equalsIgnoreCase("FULL")) {
                // With 'FULL' we keep all dimensions
                dim = width * height * inputDepth;
            } else {
                // We keep the specified number of dimensions
                dim = Integer.parseInt(readElement(unitEl, "dimensions"));
            }

            // Parse the 'layer' text
            String layerClassName = readElement(unitEl, "layer");

            // Create the unit with specified parameters
            unit = new PCAAutoEncoder(
                    width,
                    height,
                    inputDepth,
                    dim,
                    layerClassName
            );
        }

        if (type.equalsIgnoreCase("LDA")) {
            // Parse the 'dimensions' text
            int dim = Integer.parseInt(readElement(unitEl, "dimensions"));

            // Parse the 'layer' text
            String layerClassName = readElement(unitEl, "layer");

            // Create the unit with specified parameters
            unit = new LDAAutoEncoder(
                    width,
                    height,
                    inputDepth,
                    dim,
                    layerClassName
            );
        }

        if (type.equalsIgnoreCase("KMeans")) {
            error("KMeans unit no longer supported after refactor");
        }
        
        if (unit==null) {
            error("unknown unit type: "+type);
        }
        
        scae = process(scae, unit, ox, oy);

        script.scae.put(id, scae);
        return "";
    }
    
    /**
     * Returns an ID given in an element.
     * @param e the element
     * @return an id
     */
    protected String readId(Element e) {
        return readAttribute(e, "id");
    }
    
    /**
     * Creates an scae or adds a layer to it.
     * @param scae can be null in CreateStackedAE
     * @param unit autoencoder to use for the first/new layer
     * @param ox offset-x when convolving
     * @param oy offset-y when convolving
     * @return the SCAE
     */
    protected SCAE process(SCAE scae, AutoEncoder unit, int ox, int oy) {
        scae = new SCAE(
                unit,
                ox,
                oy
        );
        return scae;
    }
    
    /**
     * Either returns an SCAE (when adding layers) or null (when creating one)
     * @param id of the SCAE
     * @return null or a reference
     */
    protected SCAE getAE(String id) {
        return null;
    }
    
    

    @Override
    public String tagName() {
        return "create-scae";
    }
    
}
