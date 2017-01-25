package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.Dataset;
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
public class Remove extends AbstractCommand {

    public Remove(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String ref = readAttribute(element, "ref");

        SCAE scae = script.scae.get(ref);
        if (scae != null) {
            script.println("Removing SCAE: " + ref);
            script.scae.remove(ref);
            return "";
        }

        Classifier classifier = script.classifiers.get(ref);
        if (classifier != null) {
            script.println("Removing CLASSIFIER: " + ref);
            script.classifiers.remove(ref);
            return "";
        }

        Dataset ds = script.datasets.get(ref);
        if (ds != null) {
            script.println("Removing DATASET: " + ref);
            script.datasets.remove(ref);
            return "";
        }

        return "";
    }

    @Override
    public String tagName() {
        return "remove";
    }

}
