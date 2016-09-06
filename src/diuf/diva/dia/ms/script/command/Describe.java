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
import diuf.diva.dia.ms.ml.mlnn.MLNN;
import diuf.diva.dia.ms.ml.ae.scae.Convolution;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.ml.layer.NeuralLayer;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Dataset;
import org.jdom2.Element;

/**
 * Describes something. Described in the doc :-)
 * @author Mathias Seuret, Michele Alberti
 */
public class Describe extends AbstractCommand {

    public Describe(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String ref = readAttribute(element, "ref");

        SCAE scae = script.scae.get(ref);
        Dataset ds = script.datasets.get(ref);
        Classifier classifier = script.classifiers.get(ref);

        if (scae == null && ds == null && classifier == null) {
            error("cannot find an element with the ID "+ref);
        }

        if (scae != null) {
            describe(scae);
        }

        if (ds != null) {
            describe(ds);
        }

        if (classifier != null) {
            switch (classifier.name()) {
                case "AEClassifier":
                    describe((AEClassifier) classifier);
                    break;

                default:
                    error("invalid classifier type " + classifier.name());
            }
        }

        return "";
    }
    
    protected void describe(SCAE scae) {
        if (scae==null) {
            return;
        }
        script.println("Stacked Convolution AutoEncoder");
        script.println("Number of layers: " + scae.getLayers().size());
        for (int n=0; n<scae.getLayers().size(); n++) {
            Convolution convo = scae.getLayer(n);
            script.println("Layer " + n + ":");
            script.println("\tInput size: " + convo.getInputPatchWidth() + "x" + convo.getInputPatchHeight() + "x" + convo.getInputPatchDepth());
            script.println("\tPatch size: " + convo.getBase().getInputWidth() + "x" + convo.getBase().getInputHeight());
            script.println("\tOutput size: " + convo.getOutput().getWidth() + "x" + convo.getOutput().getHeight() + "x" + convo.getOutput().getDepth());
            script.println("\tAutoencoder: " + convo.getBase().getClass().getSimpleName());
        }
    }
    
    protected void describe(Dataset ds) {
        script.println("Dataset");
        script.println("Number of documents: " + ds.size());
        script.println("Used colorspace: " + script.colorspace);
        for (int n=0; n<ds.size(); n++) {
            DataBlock db = ds.get(n);
            script.println("\tDocument " + (n + 1) + ": " + db.getWidth() + "x" + db.getHeight() + "x" + db.getDepth());
        }
    }
    
    protected void describe(AEClassifier aec) {
        script.println("Classifier - neural network on top of an autoencoder");
        describe(aec.getSCAE());
        MLNN nn = aec.getMLNN();
        script.println("Multilayer Neural Network:");
        for (int l=0; l<nn.getLayersCount(); l++) {
            NeuralLayer nl = nn.getLayer(l);
            script.println("\tLayer " + (l + 1) + ": " + nl.getOutputSize() + " neurons");
        }
    }

    @Override
    public String tagName() {
        return "describe";
    }
    
}
