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

package diuf.diva.dia.ms.ml.ae.scae;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.util.DataBlock;

import java.io.*;
import java.util.ArrayList;

/**
 * Stacked Convolution Denoising AutoEncoder. Take note that the convolution
 * offset is set when creating the convolution layer.
 * @author Mathias Seuret, Michele Alberti
 */
public class SCAE implements Serializable {

    /**
     * The different layers of the autoencoder.
     */
    public ArrayList<Convolution> stages = new ArrayList<>();
    /**
     * A direct reference to the top-layer.
     */
    public Convolution top;
    /**
     * A direct reference to the base-layer.
     */
    public Convolution base;
    /**
     * The array in which the features are concatenated.
     */
    public float[] featureVector;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates a stacked autoencoder. It takes as input an autoencoder and
     * the offset to use in the first layer when adding a second one.
     * @param ae first autoencoder
     * @param oX offset x
     * @param oY offset y
     */
    public SCAE(AutoEncoder ae, int oX, int oY) {
        assert (ae!=null);
        assert (oX>=1);
        assert (oY>=1);

        stages.add(
                new Convolution(ae, 1, 1, oX, oY)
        );
        top  = stages.get(0);
        base = stages.get(0);
        
        setInput(
                new DataBlock(
                        getInputPatchWidth(),
                        getInputPatchHeight(),
                        getInputPatchDepth()
                )
        );
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Computing
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Encodes the layers one after another.
     * @return the output of the top values, assuming that there's only one array
     */
    public float[] forward() {
        for (Convolution convo : stages) {
            convo.encode();
        }

        return top.getOutput().getValues(0, 0);
    }

    /**
     * Decodes the layers one after another.
     */
    public void backward() {
        for (int s = stages.size() - 1; s >= 0; s--) {
            stages.get(s).rebuildInput(s != 0);
        }
    }

    /**
     * Transmits clean data to the top layer. It'll fail if that layer isn't a
     * denoising autoencoder.
     */
    public void grabClean() {
        throw new UnsupportedOperationException(
                "Denoising autoencoders needs to be reimplemented in this "
                        + "new version of N-light-N."
            );
        }

    /**
     * Trains the top-layer on the current sample.
     * @return some kind of error value
     */
    public float train() {
        for (int s=0; s<stages.size()-1; s++) {
            stages.get(s).encode();
        }
        return top.train();
    }

    /**
     * Trains the denoising top layer - fails if this layer is not denoising.
     * @param clean clean training data
     * @param noisy noisy training data
     * @param posX position x of the sample
     * @param posY position y of the sample
     * @return the reconstruction error
     */
    public float trainDenoising(DataBlock clean, DataBlock noisy, int posX, int posY) {
        assert (clean!=null);
        assert (noisy!=null);
        assert (clean.getWidth() == noisy.getWidth() && clean.getHeight() == noisy.getHeight());
        assert (clean.getWidth() > posX + getInputPatchWidth());
        assert (clean.getHeight() > posY + getInputPatchHeight());

        if (!top.getBase().isDenoising()) {
            throw new IllegalStateException(
                    "the top layer is not denoising, cannot use trainDenoising()"
            );
        }

        setInput(
                clean,
                posX,
                posY
        );
        grabClean();

        setInput(
                noisy,
                posX,
                posY
        );

        return train();
    }

    /**
     * Train the supervised top layer - fails if this layer is not supervised
     */
    public float trainSupervised(int label) {
        if (!top.getBase().isSupervised()) {
            throw new IllegalStateException(
                    "the top layer is not supervised, cannot use trainSupervised()"
            );
        }

        for (int s = 0; s < stages.size() - 1; s++) {
            stages.get(s).encode();
        }

        return top.train(label);
    }

    /**
     * Called at the end of training. Useful for PCA and LDA especially.
     */
    public void trainingDone() {
        top.getBase().trainingDone();
    }

    /**
     * Adds a layer to the SCAE. It requires not only an autoencoder, but
     * also to know which offset should be used when convolving this layer.
     * @param base autoencoder to use
     * @param oX offset X
     * @param oY offset Y
     */
    public void addLayer(AutoEncoder base, int oX, int oY) {
        assert (base!=null);
        assert (oX>=1);
        assert (oY>=1);

        if (base.needsBinaryInput() && !top.getBase().hasBinaryOutput()) {
            throw new Error(
                    base.getClass().getSimpleName()+" requires binary inputs, "+
                            "but " + top.getBase().getClass().getSimpleName() + " has real-valued " +
                            "outputs. You could insert a Binary layer to solve this."
            );
        }

        top = new Convolution(base, 1, 1, oX, oY);
        stages.add(top);

        for (int i = stages.size() - 1; i >= 1; i--) {
            Convolution prev = stages.get(i);
            Convolution curr = stages.get(i - 1);
            curr.resize(prev.getInputPatchWidth(), prev.getInputPatchHeight());
            prev.setInput(curr.getOutput());
        }

        setInput(
                new DataBlock(
                        getInputPatchWidth(),
                        getInputPatchHeight(),
                        getInputPatchDepth()
                )
        );

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Input related
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Centers the input of the SCAE at the given position of a data block.
     *
     * @param db data block
     * @param cx  new center position x on the input
     * @param cy  new center position y on the input
     */
    public void centerInput(DataBlock db, int cx, int cy) {
        base.setInput(
                db,
                cx - getInputPatchWidth()/2,
                cy - getInputPatchHeight()/2
        );
    }
    
    /**
     * @return the input patch depth
     */
    public int getInputPatchDepth() {
        return base.getInputPatchDepth();
    }

    /**
     * @return the input patch width
     */
    public int getInputPatchWidth() {
        return base.getInputPatchWidth();
    }

    /**
     * @return the input patch height
     */
    public int getInputPatchHeight() {
        return base.getInputPatchHeight();
    }

    /**
     * Sets the input data block of the SCAE. As no coordinates are specified,
     * we will assert that the input has the same dimensions as what the SCAE
     * requires.
     *
     * @param input data block
     */
    public void setInput(DataBlock input) {
        assert (input != null);
        assert (input.getWidth() == getInputPatchWidth());
        assert (input.getHeight() == getInputPatchHeight());
        assert (input.getDepth() == getInputPatchDepth());

        base.setInput(input, 0, 0);
    }

    /**
     * Sets the input of the SCAE at the given position of a data block.
     *
     * @param db data block
     * @param x  new position x on the input
     * @param y  new position y on the input
     */
    public void setInput(DataBlock db, int x, int y) {
        assert (db != null);
        assert (x >= 0);
        assert (y >= 0);
        assert (x + getInputPatchWidth() <= db.getWidth());
        assert (y + getInputPatchHeight() <= db.getHeight());
        assert (getInputPatchDepth() == db.getDepth());

        base.setInput(db, x, y);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Output related
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @return the output depth of the autoencoder
     */
    public int getOutputDepth() {
        return top.getOutput().getDepth();
    }

    public int highestOutputIndex() {
        int opt = 1;
        for (int i=1; i<top.getOutputDepth(); i++) {
            if (top.getOutput().getValue(i, 0, 0) > top.getOutput().getValue(opt, 0, 0)) {
                opt = i;
            }
        }
        return opt;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Getters & Setters
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /***
     * @return the base of the SCAE
     */
    public Convolution getBase() {
        return base;
    }

    /***
     * @return the top of the SCAE
     */
    public Convolution getTop() {
        return top;
    }

    /**
     * @param n layer number
     * @return the n-th layer
     */
    public Convolution getLayer(int n) {
        return stages.get(n);
    }

    /**
     * @return the list of convolutions compositing the SCAE
     */
    public ArrayList<Convolution> getLayers() {
        return stages;
    }

    /**
     * @return a feature vector made out of features from all layers
     */
    public float[] getCentralMultilayerFeatures() {
        if (featureVector == null) {
            int size = 0;
            for (Convolution c : stages) {
                size += c.getOutput().getDepth();
            }
            featureVector = new float[size];
        }

        int pos = 0;
        for (Convolution c : stages) {
            c.fillFeatureVector(featureVector, pos);
            pos += c.getOutput().getDepth();
        }

        float sum = 0;
        for (float f : featureVector) {
            sum += Math.abs(f);
        }

        return featureVector;
    }

    /**
     * @return the length of the vector returned by getFeatureVector()
     */
    public int getFeatureLength() {
        if (featureVector == null) {
            int size = 0;
            for (Convolution c : stages) {
                size += c.getOutput().getDepth();
            }
            featureVector = new float[size];
        }
        return featureVector.length;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Utility
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * This is used for studying the features. It extracts them and returns
     * them in the same format as the input.
     * @return a datablock
     */
    public DataBlock extractFeatures() {
        // Counting the features, getting the size of the output block
        int nbF = top.getOutput().getDepth();
        int fW = (int)(Math.sqrt(nbF));
        int fH = fW;
        while (fW*fH<nbF) {
            fW++;
        }

        // Creating the block
        DataBlock out = new DataBlock(
                fW * (getInputPatchWidth() + 1) - 1,
                fH * (getInputPatchHeight() + 1) - 1,
                getInputPatchDepth()
        );

        DataBlock tmp = new DataBlock(getInputPatchWidth(), getInputPatchHeight(), getInputPatchDepth());
        setInput(tmp);

        int n = 0;
        for (int y=0; y<fH && n<nbF; y++) {
            for (int x=0; x<fW && n<nbF; x++) {

                top.getOutput().clear();
                top.getBase().activateOutput(n, true);
                base.setInput(tmp);
                base.getInput().clear();
                for (int s=stages.size()-1; s>=0; s--) {
                    stages.get(s).rebuildInput(true);
                }
                base.getInput().normalizeWeights();

                //tmp.normalize();

                tmp.copyTo(out, x * (getInputPatchWidth() + 1), y * (getInputPatchHeight()+1));
                n++;
            }
        }

        out.normalizeWeights();

        for (int x=0; x<out.getWidth(); x++) {
            for (int y=0; y<out.getHeight(); y++) {
                for (int z=0; z<out.getDepth(); z++) {
                    if (out.getValue(z, x, y)<-1) {
                        out.setValue(z, x, y, -1);
                    } else if (out.getValue(z, x, y)>1) {
                        out.setValue(z, x, y, 1);
                    }
                }
            }
        }

        return out;
    }

    /**
     * Removes some features from the top-layer. Note that not all kinds of
     * layer support this feature.
     * @param number list of layers to remove, starting from 0
     */
    public void deleteFeatures(int... number) {
        top.deleteFeatures(number);
        featureVector = null;
    }

    /**
     * Saves the SCAE to a binary file.
     * @param fileName file name
     * @throws IOException if the file cannot be written to
     */
    public void save(String fileName) throws IOException {
        // Check whether the path is existing, if not create it
        File file = new File(fileName);
        if (!file.isDirectory()) {
            file = file.getParentFile();
        }
        if (file!=null && !file.exists()) {
            file.mkdirs();
        }

        ObjectOutputStream oop = new ObjectOutputStream(new FileOutputStream(fileName));
        // Dummy input
        setInput(new DataBlock(getInputPatchWidth(), getInputPatchHeight(), getInputPatchDepth()));
        oop.writeObject(this);
        oop.close();
    }

    /**
     * Loads a SCAE from a file.
     * @param fileName file name
     *
     * @return an instance of SCAE
     * @throws IOException if the file cannot be read
     * @throws ClassNotFoundException if the
     */
    public static SCAE load(String fileName) throws IOException, ClassNotFoundException {
        SCAE scae;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(fileName))) {
            scae = (SCAE) ois.readObject();
        }
        return scae;
    }

    @Override
    public String toString() {
        String res = "(";
        for (int i=0; i<stages.size(); i++) {
            if (i!=0) {
                res = res+" | ";
            }
            res = res + stages.get(i).toString();
        }
        return res+")";
    }
}
