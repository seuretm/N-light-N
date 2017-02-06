/*****************************************************
  Training N-light-N on MNIST
  
  -------------------
  Author:
  2016 by Mathias Seuret <mathias.seuret@unifr.ch>
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

package mnist.noconv;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.StandardAutoEncoder;
import diuf.diva.dia.ms.ml.ae.ffcnn.FFCNN;
import diuf.diva.dia.ms.ml.ae.scae.SCAE;
import java.io.IOException;
import java.util.Collections;

/**
 * Note that running this on the full MNIST dataset will require approximately
 * 1.9 GB of memory. It is advised to start the application with the following
 * virtual machine option:
 *  -Xmx4G
 * For additional information about the MNIST dataset, please read
 *  http://yann.lecun.com/exdb/mnist/
 * @author Mathias Seuret
 */
public class MNISTNoConv {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        // Loading the training data
        MNISTDataset train = new MNISTDataset(
                "data/train-images.idx3-ubyte",
                "data/train-labels.idx1-ubyte",
                60000 // note: for quick tests you can load less than the 60k images
        );
        
        // Loading the testdata
        MNISTDataset test = new MNISTDataset(
                "data/t10k-images.idx3-ubyte",
                "data/t10k-labels.idx1-ubyte"
        );
        
        // The following part if the creating of the neural network. First,
        // an Auto-Encoder is created. It takes as input a patch of 28x28 pixels
        // with 1 channel, and has 300 neurons. NeuralLayer indicates we want
        // a soft-sign activation function.
        AutoEncoder ae = new StandardAutoEncoder(28, 28, 1, 300, "NeuralLayer");
        
        // The Auto-Encoder is put within an SCAE, or stacked convolutional
        // auto-encoder. Note that after this, it would be possible to train
        // the SCAE on unlabelled data. In order to have the same architecture
        // as presented on the MNIST website, no pretraining is done here.
        // As there will be a single layer in the SCAE, the two last parameters
        // of the constructor can be ignored.
        SCAE scae = new SCAE(ae, 1, 1);
        
        // And finally, we turn the SCAE into a classifier by adding on top a
        // 10 neurons classification layer. The FFCNN, or feed forward
        // convolutional neural network, can then be trained.
        FFCNN ffcnn = new FFCNN(scae, "NeuralLayer", 10);
        
        // Then, the FFCNN is trained for 20 epochs. With 20 epochs, an accuracy
        // similar to the one reported on the MNIST website should be reached.
        // More epochs however will improve this accuracy.
        for (int epoch=0; epoch<20; epoch++) {
            System.out.println("Epoch "+(epoch+1));
            
            // It is good practice to shuffle data before each epoch
            Collections.shuffle(train.digits);
            
            // Mistakes counter, used for measuring accuracy
            int mistakes = 0;
            
            // Used for measuring training speed
            long trainingStart = System.currentTimeMillis();
            
            // Iteration over the training data
            for (MNISTDataBlock db : train.digits) {
                // We set the location (top-left corner) of the FFCNN on the
                // image being processed
                ffcnn.setInput(db, 0, 0);
                
                // And compute the expected class
                ffcnn.compute();
                
                // We can compare it to the real label
                if (ffcnn.getOutputClass(false)!=db.label) {
                    mistakes++;
                }
                
                // And set for each output neuron the value that we would have
                // wanted, i.e., 1 for the correct class, 0 for the other classes
                for (int d=0; d<10; d++) {
                    ffcnn.setExpected(d, (db.label==d) ? 1 : 0);
                }
                
                // Backpropagation of the errors
                ffcnn.backPropagate();
                
                // And last step of the training, applying what was backpropagated
                // to update the weights. Note that to use mini-batches, you should
                // call learn() every few images.
                ffcnn.learn();
            }
            float trainingTime = (System.currentTimeMillis() - trainingStart) / 1000.0f;
            
            // Output results
            System.out.println("  "+mistakes+" training errors");
            System.out.println("  "+(100.0f*mistakes)/train.digits.size()+"%");
            System.out.println("  "+(train.digits.size()/trainingTime)+" images/second");
            
            // Computation of the accuracy on test data
            mistakes = 0;
            
            // Used for measuring classification speed
            long classificationStart = System.currentTimeMillis();
            
            // Iterating over test data...
            for (MNISTDataBlock db : test.digits) {
                // Setting the input, computing output, counting mistakes
                ffcnn.setInput(db, 0, 0);
                ffcnn.compute();
                if (ffcnn.getOutputClass(false)!=db.label) {
                    mistakes++;
                }
            }
            
            // Output results
            float classificationTime = (System.currentTimeMillis() - classificationStart) / 1000.0f;
            System.out.println("  "+mistakes+" test errors");
            System.out.println("  "+(100.0f*mistakes)/test.digits.size()+"%");
            System.out.println("  "+(test.digits.size()/classificationTime)+" images/second");
            
        }
        
    }
    
}
