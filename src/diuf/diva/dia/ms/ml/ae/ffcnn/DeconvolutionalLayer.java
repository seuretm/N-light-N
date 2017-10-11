/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package diuf.diva.dia.ms.ml.ae.ffcnn;

import diuf.diva.dia.ms.ml.ae.AutoEncoder;
import diuf.diva.dia.ms.ml.ae.StandardAutoEncoder;
import diuf.diva.dia.ms.util.DataBlock;
import java.io.Serializable;

/**
 *
 * @author Mathias Seuret
 */
public class DeconvolutionalLayer implements ConvolutionalLayer, Serializable {
    protected int patchWidth;
    protected int patchHeight;
    protected int inputWidth;
    protected int inputHeight;
    protected int inputDepth;
    protected int outDepth;
    
    AutoEncoder[][] base;
    DataBlock input;
    DataBlock error;
    DataBlock prevError;
    DataBlock output;
    
    public DeconvolutionalLayer(ConvolutionalLayer previousLayer, String layerType, int patchWidth, int patchHeight, int outputDepth) {
        if (previousLayer.getOutput().getWidth()!=1 || previousLayer.getOutput().getHeight()!=1) {
            DataBlock db = previousLayer.getOutput();
            throw new Error("Deconvolutional layers require for now 1x1xN inputs, not "+db.getWidth()+"x"+db.getHeight()+"x"+db.getDepth());
        }
        
        this.patchWidth = patchWidth;
        this.patchHeight = patchHeight;
        base = new AutoEncoder[patchWidth][patchHeight];
        error = new DataBlock(patchWidth, patchHeight, outputDepth);
        output = new DataBlock(patchWidth, patchHeight, outputDepth);
        
        inputWidth = 1;
        inputHeight = 1;
        inputDepth = previousLayer.getOutput().getDepth();
        outDepth = outputDepth;
        
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y] = new StandardAutoEncoder(1, 1, previousLayer.getOutput().getDepth(), outputDepth, layerType);
                base[x][y].setInput(previousLayer.getOutput(), 0, 0);
                base[x][y].setPrevError(previousLayer.getError());
                base[x][y].setError(error);
                base[x][y].setOutput(output, 0, 0);
            }
        }
        
        System.out.println("Deconvolutional layer created");
    }
    
    @Override
    public void setInput(DataBlock db, int posX, int posY) {
        input = db;
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].setInput(db, posX, posY);
            }
        }
    }
    
    public void setOutput(DataBlock db, int posX, int posY) {
        output = db;
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].setOutput(db, posX+x, posY+y);
            }
        }
    }

    @Override
    public int getInputWidth() {
        return inputWidth;
    }

    @Override
    public int getInputHeight() {
        return inputHeight;
    }

    @Override
    public DataBlock getInput() {
        return input;
    }

    @Override
    public void resize(int outWidth, int outHeight) {
        if (outWidth%patchWidth!=0) {
            throw new Error("Wrong width, "+outWidth+" not multiple of "+patchWidth);
        }
        if (outHeight%patchHeight!=0) {
            throw new Error("Wrong height, "+outHeight+" not multiple of "+patchHeight);
        }
        inputWidth = outWidth / patchWidth;
        inputHeight = outHeight / patchHeight;
        
        error = new DataBlock(outWidth, outHeight, outDepth);
        output = new DataBlock(outWidth, outHeight, outDepth);
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].setInput(input, 0, 0);
                base[x][y].setOutput(output, 0, 0);
                base[x][y].setError(error);
            }
        }
    }

    @Override
    public void setPrevError(DataBlock db) {
        prevError = db;
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].setPrevError(db);
            }
        }
    }

    @Override
    public DataBlock getPrevError() {
        return prevError;
    }

    @Override
    public void addError(int x, int y, int z, float e) {
        error.addValue(z, x, y, e);
    }

    @Override
    public void clearError() {
        for (int ix=0; ix<inputWidth; ix++) {
            for (int iy=0; iy<inputHeight; iy++) {
                setInput(input, ix, iy);
                setOutput(output, ix*patchWidth, iy*patchHeight);
                for (int ox=0; ox<patchWidth; ox++) {
                    for (int oy=0; oy<patchHeight; oy++) {
                        base[ox][oy].clearError();
                    }
                }
            }
        }
        error.clear();
    }

    @Override
    public void setExpected(int z, float ex) {
        setExpected(0, 0, 0, ex);
    }

    @Override
    public void setExpected(int x, int y, int z, float ex) {
        float e = output.getValue(z, x, y) - ex;
        addError(x, y, z, e);
    }

    @Override
    public void setExpectedClass(int x, int y, int cNum) {
        int px = x / patchWidth;
        int py = y / patchHeight;
        setInput(input, px, py);
        setOutput(output, px, py);
        setPrevError(prevError);
        base[x % patchWidth][y % patchHeight].getEncoder().setExpectedClass(cNum);
    }

    @Override
    public DataBlock getError() {
        return error;
    }

    @Override
    public void learn() {
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].learn();
            }
        }
    }

    @Override
    public float backPropagate() {
        float e = 0;
        for (int ix=0; ix<inputWidth; ix++) {
            for (int iy=0; iy<inputHeight; iy++) {
                setInput(input, ix, iy);
                setOutput(output, ix*patchWidth, iy*patchHeight);
                setPrevError(prevError);
                for (int ox=0; ox<patchWidth; ox++) {
                    for (int oy=0; oy<patchHeight; oy++) {
                        e += base[ox][oy].backPropagate();
                    }
                }
            }
        }
        for (int x=0; x<inputWidth; x++) {
            for (int y=0; y<inputHeight; y++) {
                for (int z=0; z<inputDepth; z++) {
                    prevError.setValue(z, x, y, prevError.getValue(z, x, y)/(patchWidth*patchHeight));
                }
            }
        }
        error.clear();
        return e / (patchWidth*patchHeight*inputWidth*inputHeight);
    }

    @Override
    public void compute() {
        for (int ix=0; ix<inputWidth; ix++) {
            for (int iy=0; iy<inputHeight; iy++) {
                setInput(input, ix, iy);
                setOutput(output, ix*patchWidth, iy*patchHeight);
                for (int x=0; x<patchWidth; x++) {
                    for (int y=0; y<patchHeight; y++) {
                        base[x][y].encode();
                    }
                }
            }
        }
    }

    @Override
    public DataBlock getOutput() {
        return output;
    }

    @Override
    public AutoEncoder getAutoEncoder(int x, int y) {
        return base[x % patchWidth][y % patchHeight];
    }

    @Override
    public int getXoffset() {
        return 1;
    }

    @Override
    public int getYoffset() {
        return 1;
    }

    @Override
    public float getLearningSpeed() {
        return base[0][0].getEncoder().getLearningSpeed();
    }

    @Override
    public void setLearningSpeed(float s) {
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].setLearningSpeed(s);
            }
        }
    }

    @Override
    public void clearGradient() {
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].clearGradient();
            }
        }
    }

    @Override
    public void startTraining() {
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].startTraining();
            }
        }
    }

    @Override
    public void stopTraining() {
        for (int x=0; x<patchWidth; x++) {
            for (int y=0; y<patchHeight; y++) {
                base[x][y].stopTraining();
            }
        }
    }

    @Override
    public boolean isTraining() {
        return base[0][0].isTraining();
    }
    
}
