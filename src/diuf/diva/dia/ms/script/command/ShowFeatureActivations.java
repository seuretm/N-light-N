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

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import diuf.diva.dia.ms.script.XMLScript;
import diuf.diva.dia.ms.util.DataBlock;
import diuf.diva.dia.ms.util.Image;
import org.jdom2.Element;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;

/**
 *
 * @author ms
 */
public class ShowFeatureActivations extends AbstractCommand {

    public ShowFeatureActivations(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        String doc = readElement(element, "document");
        String ref = readAttribute(element, "ref");
        String res = readElement(element, "result");

        SCAE scae = script.scae.get(ref);
        Image img = new Image(doc);
        img.convertTo(script.colorspace);
        DataBlock idb = new DataBlock(img);
        
        BufferedImage bi = new BufferedImage(idb.getWidth(), idb.getHeight(), BufferedImage.TYPE_INT_RGB);

        int dx = scae.getInputPatchWidth() / 2;
        int dy = scae.getInputPatchHeight() / 2;
        for (int i=0; i<scae.getOutputDepth(); i++) {
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;
            for (int x = 0; x < idb.getWidth() - scae.getInputPatchWidth(); x++) {
                int mm = bi.getWidth() - scae.getInputPatchWidth();
                System.out.println(i+"/"+scae.getOutputDepth()+": "+(float)x/mm*100+"%");
                for (int y = 0; y < idb.getHeight() - scae.getInputPatchHeight(); y++) {
                    scae.setInput(idb, x, y);
                    float act = scae.forward()[i];
                    float hue = (act+1)/2 * 0.4f; // Hue (note 0.4 = Green, see huge chart below)
                    bi.setRGB(x+dx, y+dy, Color.getHSBColor(hue, 0.9f, 0.9f).getRGB());
                    
                    if (act<min) {
                        min = act;
                    }
                    if (act>max) {
                        max = act;
                    }
                }
            }
            ImageIO.write(bi, "png", new File(res+"-"+i+".png"));
            //System.out.println("Reconstruction range: ["+min+", "+max+"]");
            //System.out.println("Class: "+scae.getLayer(scae.getLayers().size()-1).getAE().getClass().getSimpleName());
        }
        
        return "";
    }

    @Override
    public String tagName() {
        return "show-feature-activations";
    }
    
}
