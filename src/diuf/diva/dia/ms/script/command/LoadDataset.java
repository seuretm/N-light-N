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
import diuf.diva.dia.ms.util.BiDataBlock;
import diuf.diva.dia.ms.util.Dataset;
import diuf.diva.dia.ms.util.Image;
import diuf.diva.dia.ms.util.NoisyDataset;
import org.jdom2.Element;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.List;

/**
 * This class loads a dataset and stores it in memory
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class LoadDataset extends AbstractCommand {
    
    public LoadDataset(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        if (element.getChild("folder")!=null && element.getChild("buffered")==null) {
            script.println("Loading dataset: " + readAttribute(element, "id"));
            return loadDataset(element);
        }
        
        if (element.getChild("folder")!=null && element.getChild("buffered")!=null) {
            script.println("Loading buffered dataset");
            return loadBufferedDataset(element);
        }
        
        if (element.getChild("clean-folder")!=null && element.getChild("noisy-folder")!=null) {
            script.println("Loading noisy dataset");
            return loadNoisyDataset(element);
        }
        
        error("either use <folder>, or both <clean-folder> and <noisy-folder> tags");
        
        return "";
        
    }

    /*
     * Load a normal dataset (not noisy and not buffered!)
     */
    private String loadDataset(Element element) throws Exception {
        String id     = readAttribute(element, "id");
        String folder = readElement(element, "folder");
        int limit     = Integer.parseInt(readElement(element, "size-limit"));
        if (limit==0) {
            limit = Integer.MAX_VALUE;
        }
        
        Dataset ds = new Dataset(folder, script.colorspace, limit);
        
        script.datasets.put(id, ds);
        
        return "";
    }
    
    private String loadBufferedDataset(Element element) throws Exception {
        if (script.colorspace!=Image.Colorspace.RGB) {
            throw new Error("<buffered/> allowed only when the RGB colorspace is used");
        }
        
        String id     = readAttribute(element, "id");
        String folder = readElement(element, "folder");
        int limit     = Integer.parseInt(readElement(element, "size-limit"));
        if (limit==0) {
            limit = Integer.MAX_VALUE;
        }
        
        File ff = new File(folder);
        String[] lst = ff.list();
        for (int i=0; i<lst.length; i++) {
            int j = (int)(Math.random()*lst.length);
            String s = lst[i];
            lst[i] = lst[j];
            lst[j] = s;
        }
        
        float[] scale = new float[]{1};
        if (element.getChild("scales")!=null) {
            List<Element> s = element.getChild("scales").getChildren("scale");
            scale = new float[s.size()];
            for (int i=0; i<s.size(); i++) {
                scale[i] = Float.parseFloat(s.get(i).getText());
            }
        }
        
        Dataset ds = new Dataset(script.colorspace);
        
        for (int i=0; i<limit && i<lst.length; i++) {
            if (lst[i].equals(".DS_Store")) {
                continue;
            }
            script.println("Loading " + folder + File.separator + lst[i]);
            BufferedImage bi = ImageIO.read(
                    new File(
                            folder+File.separator+lst[i]
                    )
            );
            for (float aScale : scale) {
                BufferedImage b = resize(bi, aScale);
                ds.add(new BiDataBlock(b));
            }
        }
        
        script.datasets.put(id, ds);
        
        return "";
    }
    
    private String loadNoisyDataset(Element element) throws IOException {
        String id     = readAttribute(element, "id");
        String cfolder = readElement(element, "clean-folder");
        String nfolder = readElement(element, "noisy-folder");
        
        if (script.datasets.containsKey(id)) {
            script.datasets.remove(id);
        }
        
        int limit = Integer.parseInt(readElement(element, "size-limit"));
        if (limit==0) {
            limit = Integer.MAX_VALUE;
        }
        
        NoisyDataset ds = new NoisyDataset(cfolder, nfolder, script.colorspace, limit);
        
        script.noisyDataSets.put(id, ds);
        
        return "";
    }

    @Override
    public String tagName() {
        return "load-dataset";
    }
    
    private static BufferedImage resize(BufferedImage img, float ratio) {
        int w = img.getWidth();
        int h = img.getHeight();
        
        int newW = (int)Math.ceil(ratio*w);
        int newH = (int)Math.ceil(ratio*h);
        
        BufferedImage dimg = new BufferedImage(newW, newH, img.getType());
        Graphics2D g = dimg.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
        RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, newW, newH, 0, 0, w, h, null);
        g.dispose();
        return dimg;
    }
    
}
