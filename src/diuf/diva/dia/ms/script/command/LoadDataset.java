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
import diuf.diva.dia.ms.util.*;
import org.jdom2.Element;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

/**
 * This class loads a dataset and stores it in memory
 *
 * @author Mathias Seuret, Michele Alberti
 */
public class LoadDataset extends AbstractCommand {

    /**
     * Constructor of the class.
     * @param script which creates the command
     */
    public LoadDataset(XMLScript script) {
        super(script);
    }

    public static Dataset<DataBlock> loadDataset(String folder, int limit, boolean buffered) {
        Dataset<DataBlock> ds = new Dataset(Dataset.TYPE.NORMAL);
        ds.loadDataBlocks(folder, limit, buffered);
        return ds;
    }


    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PRIVATE
    ///////////////////////////////////////////////////////////////////////////////////////////////

    public static Dataset<GroundTruthDataBlock> loadGroundTruthDataset(String folder, String groundTruth, int limit, boolean buffered) {

        if (!buffered) {
            throw new RuntimeException("GroundTruthDataset need to be buffered!");
        }

        Dataset<DataBlock> ds = loadDataset(folder, limit, true);
        Dataset<DataBlock> gt = loadDataset(groundTruth, limit, true);

        assert (ds.size() == gt.size());

        Dataset<GroundTruthDataBlock> dataset = new Dataset<>(Dataset.TYPE.GT);

        // For all DataBlocks in the dataset
        for (int i = 0; i < ds.size(); i++) {
            // Get next DataBlock
            DataBlock db = gt.get(i);
            int[][] values = new int[db.getWidth()][db.getHeight()];
            // Read the gt values from the blue channel
            for (int x = 0; x < db.getWidth(); x++) {
                for (int y = 0; y < db.getHeight(); y++) {
                    values[x][y] = Math.round((db.getValues(x, y)[db.getDepth() - 1] + 1) * 255.0f / 2.0f);
                }
            }
            // Create and add the new GroundTruthDataBlock
            dataset.add(new GroundTruthDataBlock((BiDataBlock) ds.get(i), values));
        }

        return dataset;
    }

    private static BufferedImage resize(BufferedImage img, float ratio) {
        int w = img.getWidth();
        int h = img.getHeight();

        int newW = (int) Math.ceil(ratio * w);
        int newH = (int) Math.ceil(ratio * h);

        BufferedImage dimg = new BufferedImage(newW, newH, img.getType());
        Graphics2D g = dimg.createGraphics();
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(img, 0, 0, newW, newH, 0, 0, w, h, null);
        g.dispose();
        return dimg;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // PUBLIC STATIC
    ///////////////////////////////////////////////////////////////////////////////////////////////

    @Override
    public String execute(Element element) throws Exception {
        if (element.getChild("folder") != null) {
            XMLScript.println("Loading dataset: " + readAttribute(element, "id"));
            if (element.getChild("filenamebased") != null) {
                // No longer supported!
                throw new RuntimeException("no longer supported, see GroundTruthDataBlock");
                //return loadFileNameBasedDataset(element);
            } else {
                String id = readAttribute(element, "id");
                String folder = readElement(element, "folder");
                int limit = Integer.parseInt(readElement(element, "size-limit"));
                if (limit == 0) {
                    limit = Integer.MAX_VALUE;
                }

                if (element.getChild("groundTruth") != null) {
                    script.datasets.put(id, loadGroundTruthDataset(folder, readElement(element, "groundTruth"), limit, element.getChild("buffered") != null));
                } else {
                    loadDataset(id, folder, limit, element.getChild("buffered") != null);
                }
                return "";
            }
        }

        if (element.getChild("clean-folder") != null && element.getChild("noisy-folder") != null) {
            XMLScript.println("Loading noisy dataset");
            return loadNoisyDataset(element);
        }

        error("either use <folder>, or both <clean-folder> and <noisy-folder> tags");

        return "";

    }

    private void loadDataset(String id, String folder, int limit, boolean buffered) {
        script.datasets.put(id, loadDataset(folder, limit, buffered));
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

        NoisyDataset ds = new NoisyDataset(cfolder, nfolder, limit);

        script.noisyDataSets.put(id, ds);

        return "";
    }

    @Override
    public String tagName() {
        return "load-dataset";
    }

}


