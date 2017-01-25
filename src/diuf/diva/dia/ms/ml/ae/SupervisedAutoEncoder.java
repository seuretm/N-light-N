package diuf.diva.dia.ms.ml.ae;

/**
 * This interface specifies a couple of methods which are necessarily required
 * to train an auto encoder in a supervised fashion.
 * For example, the LDAAutoEncoder needs class label for being trained properly.
 *
 * @author Michele Alberti
 */
public interface SupervisedAutoEncoder {

    /**
     * Trains the auto-encoder and gives the relative label for the input examined
     *
     * @return an estimation of the reconstruction error
     */
    float train(int label);

}
