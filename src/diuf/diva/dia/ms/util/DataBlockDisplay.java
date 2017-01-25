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

package diuf.diva.dia.ms.util;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

/**
 * This is a JFrame which displays the content of a datablock, assuming
 * that it contains RGB or grayscale data.
 * @author Mathias Seuret
 */
public abstract class DataBlockDisplay extends JFrame {
    /**
     * An image on which the datablock is displayed.
     */
    BufferedImage bi;
    
    /**
     * Reference to the datablock.
     */
    DataBlock db;
    
    /**
     * Constructs a datablock display.
     * @param db datablock
     * @param title  window title
     */
    public DataBlockDisplay(DataBlock db, String title) {
        this.db = db;
        
        int w = db.getWidth();
        int h = db.getHeight();
        int r = 1;
        
        while (r*w<512 && r*h<512) {
            r++;
        }
        w*=r;
        h*=r;
        
        this.setSize(w, h);
        
        bi = new BufferedImage(db.getWidth(), db.getHeight(), BufferedImage.TYPE_INT_RGB);
        
        
        JPanel pane = new JPanel() {
            @Override
            protected void paintComponent(Graphics g) {
                g.drawImage(
                        bi,
                        0,
                        0,
                        g.getClipBounds().width,
                        g.getClipBounds().height,
                        0,
                        0,
                        bi.getWidth(),
                        bi.getHeight(),
                        null
                );
            }
        };
        add(pane);
        
        setTitle(title);
        setVisible(true);
    }
    
    /**
     * To updates the content of the datablock.
     */
    public abstract void update();
    
    /**
     * Updates the image.
     * @param db specify which datablock is used.
     */
    protected void updateImage(DataBlock db) {
        switch (db.getDepth()) {
            case 1:
                updateGrayscale(db);
                break;
            case 3:
                updateColor(db);
                break;
            default:
                throw new Error("cannot display "+db.getDepth()+"-channels datablocks");
        }
        repaint();
    }
    
    /**
     * Draws the image.
     * @param db datablock
     */
    protected void updateGrayscale(DataBlock db) {
        for (int x=0; x<db.getWidth(); x++) {
            for (int y=0; y<db.getHeight(); y++) {
                float v = (db.getValue(0, x, y) + 1) / 2;
                v = (v<0) ? 0 : v;
                v = (v>1) ? 1 : v;
                int g = (int) (255 * v);
                bi.setRGB(x, y, g | (g<<8) | (g<<16));
            }
        }
    }
    
    /**
     * Draws the image
     * @param db datablock
     */
    protected void updateColor(DataBlock db) {
        for (int x=0; x<db.getWidth(); x++) {
            for (int y=0; y<db.getHeight(); y++) {
                float v = (db.getValue(0, x, y) + 1) / 2;
                v = (v<0) ? 0 : v;
                v = (v>1) ? 1 : v;
                int r = (int) (255 * v);
                
                v = (db.getValue(1, x, y) + 1) / 2;
                v = (v<0) ? 0 : v;
                v = (v>1) ? 1 : v;
                int g = (int) (255 * v);
                
                v = (db.getValue(2, x, y) + 1) / 2;
                v = (v<0) ? 0 : v;
                v = (v>1) ? 1 : v;
                int b = (int) (255 * v);
                
                bi.setRGB(x, y, (r<<16) | (g<<8) | b);
            }
        }
    }
    
    /**
     * Pastes the image on the window.
     * @param g graphics
     */
    public void paintComponent(Graphics g) {
        g.drawImage(
                bi,
                0,
                0,
                g.getClipBounds().width,
                g.getClipBounds().height,
                0,
                0,
                bi.getWidth(),
                bi.getHeight(),
                null
        );
    }
}
