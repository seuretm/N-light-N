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

package diuf.diva.dia.ms.script;

import diuf.diva.dia.ms.ml.Classifier;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.command.*;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.Image;
import diuf.diva.dia.ms.util.NoisyDataset;
import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.regex.Matcher;


/**
 * @author Mathias Seuret
 */
public class XMLScript {
    /**
     * Maps IDS to datasets.
     */
    public final HashMap<String, Dataset> datasets = new HashMap<>();

    /**
     * Maps IDS to datasets who have their GT already provided. This is the case for CIFAR, MNIST, SVHN, ... where the GT is in the file name
     */
    public final HashMap<String, HashMap<Integer, ArrayList<DataBlock>>> fileNameBasedDatasets = new HashMap<>();

    /**
     * Maps IDs to noisy datasets.
     */
    public final HashMap<String, NoisyDataset> noisyDataSets = new HashMap<>();
    
    /**
     * Maps tag names to commands.
     */
    public final HashMap<String, AbstractCommand> commands = new HashMap<>();
    
    /**
     * Maps IDs to autoencoders.
     */
    public final HashMap<String, SCAE> scae = new HashMap<>();
    
    /**
     * Maps words to definitions.
     */
    public final HashMap<String, String> definitions = new HashMap<>();
    
    /**
     * Maps IDs to classifiers.
     */
    public final HashMap<String, Classifier> classifiers = new HashMap<>();
    
    /**
     * Stores which color space is being used by
     * the script. All commands MUST take this
     * into account where needed.
     */
    public Image.Colorspace colorspace = Image.Colorspace.RGB;
    /**
     * Root element of the XML script.
     */
    protected Element root;

    /**
     * Constructs an XML script.
     * @param fname file name from which to read the XML file
     * @throws IOException if the file cannot be read
     * @throws JDOMException if the XML is not valid
     */
    public XMLScript(String fname) throws IOException, JDOMException {
        SAXBuilder builder = new SAXBuilder();
        Document xml = builder.build(new File(fname));
        root = xml.getRootElement();
        readColorspace();
        prepareCommands();
    }

    /**
     * Pretty print of the execution duration
     */
    public static void printDuration(long start) {
        long chrono = System.currentTimeMillis() - start;
        long t = chrono / 1000;

        if (t < 60) {
            System.out.println("Run time: " + t + " seconds");
        } else if (t < 3600) {
            System.out.printf("Run time: %d:%02d\n", (t / 60), (t % 60));
        } else {
            System.out.printf("Run time: %d:%02d:%02d\n", (t / 3600), ((t % 3600) / 60), (t % 60));
        }
    }
    
    /**
     * Deletes everything. Might be useful to free memory.
     */
    public void clearData() {
        datasets.clear();
        commands.clear();
        scae.clear();
        definitions.clear();
        classifiers.clear();
    }

    /**
     * Prepares instances of the different commands.
     */
    private void prepareCommands() {
        // Load & save
        addCommand(new LoadDataset(this));
        addCommand(new UnloadDataset(this));
        addCommand(new Save(this));
        addCommand(new Load(this));
        addCommand(new Remove(this));
        // Scae
        addCommand(new CreateStackedAE(this));
        addCommand(new AddLayer(this));
        addCommand(new TrainSCAE(this));
        addCommand(new TrainDecoderOnly(this));
        addCommand(new Recode(this));
        addCommand(new ShowFeatures(this));
        addCommand(new ShowFeatureActivations(this));
        addCommand(new EvaluateReconstruction(this));
        //Classifier
        addCommand(new CreateClassifier(this));
        addCommand(new TrainClassifier(this));
        addCommand(new EvaluateClassifier(this));
        // Utility
        addCommand(new DeleteFeatures(this));
        addCommand(new Beep(this));
        addCommand(new Describe(this));
        addCommand(new Define(this));
        addCommand(new Print(this));
        addCommand(new StoreResult(this));
        addCommand(new SetRandom(this));

    }

    /**
     * Modifies a string by replacing all defined words by their definitions.
     * @param in input string
     * @return a new string
     */
    public String preprocess(String in) {
        /* I pre-process three times so that I am "sure" that nested references get resolved!
         * This issue is raised by the nasty order of elements in 'definitions' determined
         * by their hash value instead of insertion order.
         */
        for (int i = 0; i < 3; i++) {
            for (String key : definitions.keySet()) {
                in = in.replaceAll(Matcher.quoteReplacement(key), definitions.get(key));
            }
        }
        return in;
    }

    /**
     * Adds a command to the script.
     * @param cmd new command
     */
    private void addCommand(AbstractCommand cmd) {
        commands.put(cmd.tagName(), cmd);
    }

    /**
     * Loads the color space from the XML.
     */
    private void readColorspace() {
        String cs = root.getAttributeValue("colorspace");
        if (cs==null) {
            return;
        }
        try {
            colorspace = Image.Colorspace.valueOf(cs);
        } catch (Exception e) {
            throw new Error(
                    "Colorspace "+cs+" does not exist. Try:\n"+
                    Arrays.toString(Image.Colorspace.values())
            );
        }
    }

    /**
     * Runs the script.
     * @return the output of the last command
     * @throws Exception in unfortunately too many cases
     */
    public String execute() throws Exception {
        String res = "";
        for (Element e : root.getChildren()) {
            String name = e.getName();
            AbstractCommand cmd = commands.get(name);
            if (cmd==null) {
                throw new Error(
                        "Cannot find command "+name
                );
            }
            String cmdRes = cmd.execute(e);
            if (res.equals("") || !cmdRes.equals("")) {
                res = cmdRes;
            }
            definitions.put("$ANS", String.valueOf(res));
        }
        return res;
    }
    
    /**
     * Prints stuff with a timestamp.
     * @param s string to print
     */
    public static void println(String s) {
        SimpleDateFormat ft = new SimpleDateFormat ("HH:mm:ss.SSS");
        System.out.println(ft.format(new Date())+": "+s);
    }

    /**
     * Prints stuff with a timestamp without the new line
     *
     * @param s string to print
     */
    public static void print(String s) {
        SimpleDateFormat ft = new SimpleDateFormat("HH:mm:ss.SSS");
        System.out.print(ft.format(new Date()) + ": " + s);
    }
}
