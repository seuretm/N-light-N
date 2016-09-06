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
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.Image;
import org.jdom2.Element;

import java.io.File;
import java.io.IOException;

/**
 *
 * @author Mathias Seuret, Alberti Michele
 */
public class Recode extends AbstractCommand {

    public Recode(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        
        if (element.getChild("file")!=null) {
            return recodeFile(element);
        }

        // Getting the SCAE
        String id = readAttribute(element, "ref");
        SCAE scae = script.scae.get(id);
        if (scae==null) {
            error("cannot autoencoder find "+id);
        }

        // Getting dataset
        String dsName = readElement(element, "dataset");
        Dataset ds = script.datasets.get(dsName);
        if (ds==null) {
            error("cannot find dataset"+id);
        }

        // Parsing output folder path
        String dst = readElement(element, "destination");

        // Check whether the path is existing, if not create it
        File file = new File(dst);
        if (!file.exists()) {
            file.mkdirs();
        }
        if (!file.isDirectory()) {
            error(dst + " is not a folder");
        }

        // Offset is default input width, but can be specified if necessary
        int offsetX = scae.getInputPatchWidth();
        if (element.getChild("offset-x") != null) {
            offsetX = Integer.parseInt(readElement(element, "offset-x"));
        }
        int offsetY = scae.getInputPatchHeight();
        if (element.getChild("offset-y") != null) {
            offsetY = Integer.parseInt(readElement(element, "offset-y"));
        }

        script.println("SCAE Starting recoding {offset:" + offsetX + "," + offsetY + "}");

        for (int n=0; n<ds.size(); n++) {
            DataBlock db = ds.get(n);
            DataBlock res = new DataBlock(db.getWidth(), db.getHeight(), db.getDepth());

            for (int x = 0; x < db.getWidth() - scae.getInputPatchWidth(); x += offsetX) {
                for (int y = 0; y < db.getHeight() - scae.getInputPatchHeight(); y += offsetY) {
                    scae.setInput(db, x, y);
                    scae.forward();
                    scae.setInput(res, x, y);
                    scae.backward();
                }
            }
            
            res.normalizeWeights();
            res.setColorspace(script.colorspace);
            res.getImage().write(dst+"/"+n+".png");
        }
        
        return "";
    }

    @Override
    public String tagName() {
        return "recode";
    }

    public static DataBlock recode(SCAE scae, DataBlock db, Image.Colorspace colorspace) {
        DataBlock res = new DataBlock(db.getWidth(), db.getHeight(), db.getDepth());
        for (int x = 0; x <= db.getWidth() - scae.getInputPatchWidth(); x += scae.getInputPatchWidth()) {
            for (int y = 0; y <= db.getHeight() - scae.getInputPatchHeight(); y += scae.getInputPatchHeight()) {
                scae.setInput(db, x, y);
                scae.forward();
                scae.setInput(res, x, y);
                scae.backward();
            }
        }

        res.normalizeWeights();
        res.setColorspace(colorspace);
        return res;
    }

    private String recodeFile(Element element) throws IOException {
        String id = readAttribute(element, "ref");
        String file = readElement(element, "file");
        String dst    = readElement(element, "destination");

        SCAE scae = script.scae.get(id);
        if (scae==null) {
            error("cannot autoencoder find "+id);
        }
        
        if (!(new File(file).exists())) {
            error("Cannot find "+file);
        }
        
        Image img = new Image(file);
        img.convertTo(script.colorspace);
        DataBlock db = new DataBlock(img);
        
        DataBlock res = new DataBlock(db.getWidth(), db.getHeight(), db.getDepth());
        for (int x = 0; x <= db.getWidth() - scae.getInputPatchWidth(); x += scae.getInputPatchWidth()) {
            for (int y = 0; y <= db.getHeight() - scae.getInputPatchHeight(); y += scae.getInputPatchHeight()) {
                scae.setInput(db, x, y);
                scae.forward();
                scae.setInput(res, x, y);
                scae.backward();
            }
        }

        res.normalizeWeights();
        res.setColorspace(script.colorspace);
        res.getImage().write(dst);
        
        return "";
    }
    
}
