package diuf.diva.dia.ms.ml;

/**
 * This class defines the basic interface standard for a classifier whic
 * allows for pre-training in the framework. Currently (September 2016),
 * the only classifier which support such thing is FFCNN because it can
 * feature LDAAutoEncoder as layers. On the other hand, AEClassifier cannot
 * work with LDAAutoEncoder because it uses a MLNN which is fixed to be made
 * of NeuralLayer and does not allow the employment of AutoEncoders of any kind.
 *
 * @author Michele Alberti
 */
public interface PreTrainable {

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Setting input
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method makes sure that all the AEs get the training sample into their training dataset.
     * Before calling this is mandatory that the input is set correctly into the classifier.
     */
    void addTrainingSample(int correctClass);

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Learning
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * This method must be called at the end of training
     */
    void trainingDone();

    /**
     * This method returns whether the classifier has already been pre-trained or not
     */
    boolean isTrained();

}
