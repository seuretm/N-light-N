/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.ml;

/**
 * This interface must be implemented by:<br/>
 * - All elements that can be trained<br/>
 * - Untrainable elements (e.g., MaxPooler) which super-class might have trainable sub-classes.
 * @author Mathias Seuret
 */
public interface Trainable {
    /**
     * This method should be called before starting a training session.
     * Some trainable elements might need to set up data 
     */
    public void startTraining();
    
    /**
     * This method should be called at the end of a training session.
     * Some trainable elements might need to do some clean-up.
     */
    public void stopTraining();
    
    /**
     * @return true if the element is being trained
     */
    public boolean isTraining();
}
