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
import diuf.diva.dia.ms.util.misc.ReconstructionScore;
import org.jdom2.Element;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * This command reconstructs an image and indicates how accurate the
 * reconstruction was.
 * @author Mathias Seuret
 */
public class EvaluateReconstruction extends AbstractCommand {

    /**
     * Offsets. By default they're initialised with inputPatchWidth/height
     */
    private int offsetX, offsetY;

    /**
     * Destination of the recoded file
     */
    private String dst = null;

    /**
     * index of the image being analysed in the dataset.
     */
    private int imageIndex = 1;

    /**
     * Either returns an SCAE (when adding layers) or null (when creating one)
     * @param id of the SCAE
     * @return null or a reference
     */
    public EvaluateReconstruction(XMLScript script) {
        super(script);
    }

    @Override
    public String execute(Element element) throws Exception {
        // Get the SCAE
        String ref = readAttribute(element, "ref");
        SCAE scae = script.scae.get(ref);
        if (scae == null) {
            error("cannot find " + ref + ", check the id");
        }
        assert (scae != null);

        // Parse the dataset
        String dataset = readElement(element, "dataset");
        Dataset ds = script.datasets.get(dataset);

        // Offset is default input width, but can be specified if necessary
        offsetX = scae.getInputPatchWidth();
        if (element.getChild("offset-x") != null) {
            offsetX = Integer.parseInt(readElement(element, "offset-x"));
        }
        offsetY = scae.getInputPatchHeight();
        if (element.getChild("offset-y") != null) {
            offsetY = Integer.parseInt(readElement(element, "offset-y"));
        }

        // Parse recode
        if (element.getChild("recode") != null) {
            dst = readElement(element, "recode");
            // Check whether the path is existing, if not create it
            File file = new File(dst);
            if (!file.exists()) {
                file.mkdirs();
            }
            if (!file.isDirectory()) {
                error(dst + " is not a folder");
            }
        }

        script.print("Starting SCAE Reconstruction evaluation\n");

        // For all images in dataset
        for (DataBlock db : ds) {
            float[] val = getReconstructionScore(scae, db);

            System.out.println("--- IMAGE " + imageIndex + "-------------------------");
            System.out.printf("\tEUC=%.4f : VAR=%.2f\n", val[0], val[1]);
            System.out.printf("\tSOI=%.4f : VAR=%.2f\n", val[2], val[3]);
            System.out.printf("\tNORM CORR= %.4f : VAR=%.2f\n", val[4], val[5]);
            System.out.printf("\tDELTA94= %.4f : VAR=%.2f\n", val[6], val[7]);
            System.out.printf("\tMAHALA= %.4f : VAR=%.2f\n", val[8], val[9]);
            imageIndex++;
        }

        script.print("End SCAE Reconstruction evaluation\n");

        return "";
    }

    /**
     * Encodes and decodes the input, and returns the mean distance of
     * the reconstructed pixels. The patches are offseted by the given values.
     *
     * @param input an input area
     * @return the mean different distances between input and reconstructed input
     */
    public float[] getReconstructionScore(SCAE scae, DataBlock input) throws IOException {

        DataBlock res = new DataBlock(input.getWidth(), input.getHeight(), input.getDepth());

        // Init lists
        List<Float> eucl = new ArrayList<>();
        List<Float> soid = new ArrayList<>();
        List<Float> corr = new ArrayList<>();
        List<Float> e94 = new ArrayList<>();
        List<Float> mahala = new ArrayList<>();

        // Loop over the image
        int rightBorder = (input.getWidth() - scae.getInputPatchWidth() > 0) ? input.getWidth() - scae.getInputPatchWidth() : 1;
        int bottomBorder = (input.getHeight() - scae.getInputPatchHeight() > 0) ? input.getHeight() - scae.getInputPatchHeight() : 1;

        for (int x = 0; x < rightBorder; x += offsetX) {
            for (int y = 0; y < bottomBorder; y += offsetY) {

                scae.setInput(input, x, y);
                float[] exp = scae.base.getBase().getInputArray().clone();
                scae.forward();
                scae.setInput(res, x, y);
                scae.backward();
                float[] val = scae.base.getBase().getDecoded().clone();

                // Compute the different distances
                eucl.add(ReconstructionScore.euclideanDistance(val, exp));
                soid.add(ReconstructionScore.scaleOffsetInvarDist(val, exp));
                corr.add(ReconstructionScore.normalizedCorrelation(val, exp));
                e94.add(ReconstructionScore.delta94distance(val, exp));
                mahala.add(ReconstructionScore.mahalanobisDistance(val, exp));
            }
        }

        if (dst != null) {
            res.normalizeWeights();
            res.setColorspace(script.colorspace);
            res.getImage().write(dst + "/" + imageIndex + "-recoded.png");
        }

        // Compute means and variances of all patches and return the results
        return new float[]{
                ReconstructionScore.getMean(eucl), ReconstructionScore.getVariance(eucl),
                ReconstructionScore.getMean(soid), ReconstructionScore.getVariance(soid),
                ReconstructionScore.getMean(corr), ReconstructionScore.getVariance(corr),
                ReconstructionScore.getMean(e94), ReconstructionScore.getVariance(e94),
                ReconstructionScore.getMean(mahala), ReconstructionScore.getVariance(mahala)
        };
    }

    @Override
    public String tagName() {
        return "evaluate-reconstruction";
    }


}
