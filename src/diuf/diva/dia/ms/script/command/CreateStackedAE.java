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
import diuf.diva.dia.ms.ml.ae.ffcnn.ConvolutionalLayer;
import diuf.diva.dia.ms.ml.ae.ffcnn.FFCNN;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.ObjectInputStream;

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

        /* If the <fromClassifier> tag is present, skip
         *
         */
        if (element.getChild("fromClassifier") != null) {
            script.scae.put(id, convertFromClassifier(getAE(id),readElement(element, "fromClassifier")));
            return "";
        }

        Element unitEl = element.getChild("unit");
        if (unitEl==null) {
            error("require unit tag");
        }

        String type = readElement(unitEl, "type");

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
            int hidden = Integer.parseInt(readElement(unitEl, "dimensions"));

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
        
        //TODO: add to the documentation
        if (type.equalsIgnoreCase("SPECTRAL")) {
            // Parse the 'dimensions' text
            int hidden = Integer.parseInt(readElement(unitEl, "dimensions"));

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
            
            String forward = readElement(unitEl, "forward");
            String inverse = readElement(unitEl, "inverse");

            // Create the unit with specified parameters
            unit = new SpectralAutoEncoder(
                    width,
                    height,
                    inputDepth,
                    hidden,
                    forward,
                    inverse,
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
        
        if (type.equalsIgnoreCase("POOLER")) {
            String selector =readElement(unitEl, "selector");
            unit = new Pooler(
                    width,
                    height,
                    inputDepth,
                    selector
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
     * Given a classifier (atm only FFCNN is supported!) extracts the AEs in the base
     * and creates a SCAE that is then returned
     * @param scae scae to add the layer on, if it is null then is the first layer and a new scae is created
     * @param classifierPath string of the path to the classifier *.dat file
     * @return the SCAE built with the classifier AEs
     */
    protected SCAE convertFromClassifier(SCAE scae, String classifierPath) {


        FFCNN ffcnn = null;

        // Read the *.dat file and load the content
        try {
            Object o = new ObjectInputStream(new FileInputStream(classifierPath)).readObject();
            // If it is a FFCNN classifier then assign it, otherwise throw error
            if (o instanceof FFCNN) {
                ffcnn = (FFCNN) o;
            }else{
                throw new RuntimeException("[ERROR][CreateStackedAE][convertFromClassifier] The loaded object is not of type FFCNN.");
            }
        } catch (IOException | ClassNotFoundException e) {
            e.printStackTrace();
        }

        // To avoid warnings
        assert(ffcnn!=null);

        if(scae == null) {
            // Init the scae with the unit of the first layer of the ffcnn
            ConvolutionalLayer l = ffcnn.getLayer(0);
            scae = new SCAE(l.getAutoEncoder(0,0).clone(),l.getXoffset(),l.getYoffset());
        }else{
            /* Add a layer to the SCAE corresponding to the "next" on the FFCNN (aka: if the scae
             * has 2 layers already, it copy the third layer of the FFCNN. It does not make a check
             * if it exists or not. It is your responsibility to ensure that it make sense.
             */
            ConvolutionalLayer l = ffcnn.getLayer(scae.getLayers().size());
            scae.addLayer(l.getAutoEncoder(0,0).clone(),l.getXoffset(), l.getYoffset());
        }

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
