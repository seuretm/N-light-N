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
import diuf.diva.dia.ms.ml.ae.aec.AEClassifier;
import diuf.diva.dia.ms.ml.ae.ffcnn.FFCNN;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * This class creates a classifier. It is necessary to specify which kind of classifier
 * and depending on the classifier nature it might be necessary to provide further parameters.
 * The following is a list of implemented classifiers:
 *
 * -AEClassifier
 * -FFCNN
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class CreateClassifier extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public CreateClassifier(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String type = readElement(element, "type");
        String id = readAttribute(element, "id");

        Classifier classifier = null;

        switch (type) {
            case "AEClassifier":
                classifier = createAEClassifier(element);
                break;

            case "FFCNN":
                classifier = createFFCNNClassifier(element);
                break;

            default:
                error("invalid classifier type " + type);
        }

        script.classifiers.put(id, classifier);

        return "";
    }

    /**
     * This method created an AEClassifier as specified by the parameters passed as command.
     * XML syntax to use this feature:
     *
     *  <create-classifier id="myClassifier">
     *      <type>AEClassifier</type>
     *      <scae>myScae</scae>                 // ID of the scae to use to build the classifier
     *      <neurons>int[,int]*</neurons>       // Number of neurons for each layer of classifier
     *      <classes>int</classes>              // Number of output classes
     *  </create-classifier>
     *
     */
    private AEClassifier createAEClassifier(Element element) {

        // Getting the former SCAE
        String ae = readElement(element, "scae");

        SCAE scae = script.scae.get(ae);
        if (scae == null) {
            error("cannot find " + ae);
        }

        // Parsing neurons
        int[] neurons = new int[0];
        // Only add extra neurons if the field is not empty
        if (element.getChild("neurons") != null) {
            try {
                String nc = readElement(element, "neurons");
                String[] neuronsS = nc.split(",");
                neurons = new int[neuronsS.length];
                script.println("Creating classifier with " + neuronsS.length + " hidden layers:");
                for (int n = 0; n < neuronsS.length; n++) {
                    neurons[n] = Integer.parseInt(neuronsS[n]);
                    script.println(" " + neurons[n] + " neurons in layer " + n);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("No additional neurons");
        }

        // Number of output classes
        int nbClasses = Integer.parseInt(readElement(element, "classes"));

        script.println(" " + nbClasses + " neurons in the output layer");

        // Parse the 'layer' text
        String layerClassName = "NeuralLayer";
        if (element.getChild("layer") != null) {
            layerClassName = readElement(element, "layer");
        }

        // Parse the 'ae' text
        String aeClassName = "StandardAutoEncoder";
        if (element.getChild("ae") != null) {
            aeClassName = readElement(element, "ae");
        }

        // Creating the classifier
        return new AEClassifier(scae, nbClasses, neurons);
    }

    /**
     * This method created a FFCNN classifier as specified by the parameters passed as command.
     * XML syntax to use this feature:
     * <p>
     * <create-classifier id="myClassifier">
     * <type>FFCNN</type>
     * <scae>myScae</scae>                 // ID of the scae to use to build the classifier
     * <neurons>int[,int]*</neurons>       // Number of neurons for each layer of classifier
     * <classes>int</classes>              // Number of output classes
     * </create-classifier>
     */
    private FFCNN createFFCNNClassifier(Element element) {

        // Getting the former SCAE
        String ae = readElement(element, "scae");

        SCAE scae = script.scae.get(ae);
        if (scae == null) {
            error("cannot find " + ae);
        }

        // Parsing neurons
        int[] neurons = new int[0];
        // Only add extra neurons if the field is not empty
        if (element.getChild("neurons") != null) {
            try {
                String nc = readElement(element, "neurons");
                String[] neuronsS = nc.split(",");
                neurons = new int[neuronsS.length];
                script.println("Creating classifier with " + neuronsS.length + " hidden layers:");
                for (int n = 0; n < neuronsS.length; n++) {
                    neurons[n] = Integer.parseInt(neuronsS[n]);
                    script.println(" " + neurons[n] + " neurons in layer " + n);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        } else {
            System.out.println("No additional neurons");
        }

        // Number of output classes
        int nbClasses = Integer.parseInt(readElement(element, "classes"));

        script.println(" " + nbClasses + " neurons in the output layer");

        // Parse the 'layer' text
        String layerClassName = "NeuralLayer";
        if (element.getChild("layer") != null) {
            layerClassName = readElement(element, "layer");
        }

        // Creating the classifier
        return new FFCNN(scae, layerClassName, nbClasses, neurons);

    }

    @Override
    public String tagName() {
        return "create-classifier";
    }

}
