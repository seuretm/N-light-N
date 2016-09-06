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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import static java.lang.Math.*;

/**
 * This is a class managing images.
 * @author Mathias Seuret
 */
public class Image {
    /**
     * Different available colorspaces.
     */
    public enum Colorspace {
        RGB(3), // Red, Green, Blue
        YUV(3), // Luminance, U chrominance, V chrominance
        HSV(3), // Hue, Saturation, Value
        XYZ(3), // XYZ ?
        CSSV(4), // Cos(hue), Sin(hue), Value
        CMYK(4), // Cyan, Magenta, Yellow, blacK
        CSS(3), // Cos(hue)*value, Sin(hue)*value, Saturation
        GRAYSCALE(1);
        
        public final int depth;
        Colorspace(int d) {
            depth = d;
        }
    }
    
    /**
     * Colorspace of the image
     */
    private Colorspace cs = Colorspace.RGB;
    
    /**
     * Array storing the values of the pixels. The first index corresponds
     * to the channel.
     */
    protected float[][][] pixel;
    
    /**
     * Width of the image.
     */
    protected int width;
    
    /**
     * Height of the image.
     */
    protected int height;
    
    /**
     * Loads an image from a file.
     * @param fileName of the image
     * @throws IOException if there's a reading problem
     */
    public Image(String fileName) throws IOException {
        assert ((new File(fileName)).exists());
        
        BufferedImage src = ImageIO.read(new File(fileName));
        width = src.getWidth();
        height = src.getHeight();
        pixel = new float[3][width][height];
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                int rgb = src.getRGB(x, y);
                pixel[0][x][y] = getR(rgb);
                pixel[1][x][y] = getG(rgb);
                pixel[2][x][y] = getB(rgb);
            }
        }
    }
    
    /**
     * Creates an RGB image with the given dimensions.
     * @param width of the image
     * @param height of the image
     */
    public Image(int width, int height) {
        this.width = width;
        this.height = height;
        pixel = new float[3][width][height];
    }
    
    /**
     * Creates an RGB image with the given dimensions.
     * @param width of the image
     * @param height of the image
     * @param t type of the image
     */
    public Image(int width, int height, Colorspace t) {
        this.width = width;
        this.height = height;
        this.cs = t;
        pixel = new float[t.depth][width][height];
    }
    
    /**
     * Saves the image
     * @param fileName file name (jpg or png)
     * @throws IOException if the file cannot be written
     */
    public void write(String fileName) throws IOException {
        Colorspace pType = cs;

        toRGB();
        
        BufferedImage targ = new BufferedImage(
                width,
                height,
                BufferedImage.TYPE_INT_RGB
        );
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                targ.setRGB(
                        x,
                        y,
                        getRGB(pixel[0][x][y],pixel[1][x][y],pixel[2][x][y])
                );
            }
        }
        
        String format = fileName.substring(
                fileName.length()-3,
                fileName.length()
        );
        ImageIO.write(targ, format, new File(fileName));
        
        convertTo(pType);
    }
    
    /**
     * Converts the colorspace to the given type.
     * @param t target color space 
     */
    public void convertTo(Colorspace t) {
        switch (t) {
            case RGB:
                toRGB();
                break;
            case YUV:
                toYUV();
                break;
            case HSV:
                toHSV();
                break;
            case CSSV:
                toCSSV();
                break;
            case CMYK:
                toCMYK();
                break;
            case XYZ:
                toXYZ();
                break;
            case CSS:
                toCSS();
                break;
            case GRAYSCALE:
                toGrayscale();
                break;
        }
    }
    
    /**
     * Changes the colorspace to rgb.
     */
    public void toRGB() {
        switch (cs) {
            case RGB:
                // Nothing to do
                break;
            case YUV:
                yuvToRgb();
                break;
            case HSV:
                hsvToRgb();
                break;
            case CSSV:
                cssvToHsv();
                hsvToRgb();
                break;
            case CMYK:
                cmykToRgb();
                break;
            case XYZ:
                xyzToRgb();
                break;
            case CSS:
                cssToHsv();
                hsvToRgb();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                break;
        }
        cs = Colorspace.RGB;
    }
    
    /**
     * Changes the colorspace to grayscale.
     */
    public void toGrayscale() {
        switch (cs) {
            case RGB:
                rgbToGrayscale();
                break;
            case YUV:
                yuvToRgb();
                rgbToGrayscale();
                break;
            case HSV:
                hsvToRgb();
                rgbToGrayscale();
                break;
            case CSSV:
                cssvToHsv();
                hsvToRgb();
                rgbToGrayscale();
                break;
            case CMYK:
                cmykToRgb();
                rgbToGrayscale();
                break;
            case XYZ:
                xyzToRgb();
                rgbToGrayscale();
                break;
            case CSS:
                cssToHsv();
                hsvToRgb();
                rgbToGrayscale();
                break;
            case GRAYSCALE:
                // Nothing to do
                break;
        }
        cs = Colorspace.GRAYSCALE;
    }
    
    /**
     * Changes the colorspace to YUV.
     */
    public void toYUV() {
        switch (cs) {
            case RGB:
                rgbToYuv();
                break;
            case YUV:
                // Nothing to do
                break;
            case HSV:
                hsvToRgb();
                rgbToYuv();
                break;
            case CSSV:
                cssvToHsv();
                hsvToRgb();
                rgbToYuv();
                break;
            case CMYK:
                cmykToRgb();
                rgbToYuv();
                break;
            case XYZ:
                xyzToRgb();
                rgbToYuv();
                break;
            case CSS:
                cssToHsv();
                hsvToRgb();
                rgbToYuv();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToYuv();
                break;
        }
        cs = Colorspace.YUV;
    }
    
    /**
     * Changes the colorspace to HSV.
     */
    public void toHSV() {
        switch (cs) {
            case RGB:
                rgbToHsv();
                break;
            case YUV:
                yuvToRgb();
                rgbToHsv();
                break;
            case HSV:
                // Nothing to do
                break;
            case CSSV:
                cssvToHsv();
                break;
            case CMYK:
                cmykToRgb();
                rgbToHsv();
                break;
            case XYZ:
                xyzToRgb();
                rgbToHsv();
                break;
            case CSS:
                cssToHsv();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToHsv();
                break;
        }
        cs = Colorspace.HSV;
    }
    
    /**
     * Changes the colorspace to CSSV - HSV based, with cos(H) and
     * sin(H) as the two first components.
     */
    public void toCSSV() {
        switch (cs) {
            case RGB:
                rgbToHsv();
                break;
            case YUV:
                yuvToRgb();
                rgbToHsv();
                break;
            case HSV:
                // nothing to do here
                break;
            case CSSV:
                return;
            case CMYK:
                cmykToRgb();
                rgbToHsv();
                break;
            case XYZ:
                xyzToRgb();
                rgbToHsv();
                break;
            case CSS:
                cssToHsv();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToHsv();
                break;
        }
        hsvToCssv();
        cs = Colorspace.CSSV;
    }
    
    /**
     * Changes the colorspace to CMYK.
     */
    public void toCMYK() {
        switch (cs) {
            case RGB:
                rgbToCmyk();
                break;
            case YUV:
                yuvToRgb();
                rgbToCmyk();
                break;
            case HSV:
                hsvToRgb();
                rgbToCmyk();
                break;
            case CSSV:
                cssvToHsv();
                hsvToRgb();
                rgbToCmyk();
            case CMYK:
                return;
            case XYZ:
                xyzToRgb();
                rgbToCmyk();
                break;
            case CSS:
                cssToHsv();
                hsvToRgb();
                rgbToCmyk();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToCmyk();
                break;
        }
    }
    
    /**
     * Changes the colorspace to CMYK.
     */
    public void toXYZ() {
        switch (cs) {
            case RGB:
                rgbToXyz();
                break;
            case YUV:
                yuvToRgb();
                rgbToXyz();
                break;
            case HSV:
                hsvToRgb();
                rgbToXyz();
                break;
            case CSSV:
                cssvToHsv();
                hsvToRgb();
                rgbToXyz();
            case CMYK:
                cmykToRgb();
                rgbToXyz();
                return;
            case XYZ:
                return;
            case CSS:
                cssToHsv();
                hsvToRgb();
                rgbToXyz();
                break;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToXyz();
                break;
        }
    }
    
    /**
     * Changes the colorspace to CSS.
     */
    public void toCSS() {
        switch (cs) { // to hsv first
            case RGB:
                rgbToHsv();
                break;
            case YUV:
                yuvToRgb();
                rgbToHsv();
                break;
            case HSV:
                break;
            case CSSV:
                cssvToHsv();
            case CMYK:
                cmykToRgb();
                rgbToHsv();
                return;
            case XYZ:
                xyzToRgb();
                rgbToHsv();
                return;
            case CSS:
                return;
            case GRAYSCALE:
                grayscaleToRgb();
                rgbToHsv();
                break;
        }
        hsvToCss();
    }
    
    private void rgbToYuv() {
        assert (cs==Colorspace.RGB);
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float r = (pixel[0][x][y]+1)/2;
                float g = (pixel[1][x][y]+1)/2;
                float b = (pixel[2][x][y]+1)/2;
                pixel[0][x][y] = 2*( 0.29900f*r + 0.58700f*g + 0.11400f*b) - 1;
                pixel[1][x][y] =   (-0.14713f*r - 0.28886f*g + 0.43600f*b)/0.436f;
                pixel[2][x][y] =   ( 0.61500f*r - 0.51498f*g - 0.10001f*b)/0.615f;
            }
        }
        cs = Colorspace.YUV;
    }
    
    private void yuvToRgb() {
        assert (cs==Colorspace.YUV);
        
        for (int px=0; px<width; px++) {
            for (int py=0; py<height; py++) {
                float y = (pixel[0][px][py]+1)/2;
                float u = (pixel[1][px][py])*0.436f;
                float v = (pixel[2][px][py])*0.615f;
                pixel[0][px][py] = 2*(y + 0.00000f*u + 1.13983f*v)-1;
                pixel[1][px][py] = 2*(y - 0.39465f*u - 0.58060f*v)-1;
                pixel[2][px][py] = 2*(y + 2.03211f*u + 0.00000f*v)-1;
            }
        }
        cs = Colorspace.RGB;
    }
    
    private void rgbToGrayscale() {
        assert (cs==Colorspace.RGB);
        
        float[][][] p = new float[1][width][height];
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float r    = pixel[0][x][y];
                float g    = pixel[1][x][y];
                float b    = pixel[2][x][y];
                p[0][x][y] = (r+g+b)/3;
            }
        }
        pixel = p;
        cs = Colorspace.GRAYSCALE;
    }
    
    private void grayscaleToRgb() {
        assert (cs==Colorspace.GRAYSCALE);
        
        float[][][] p = new float[3][width][height];
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float g    = pixel[0][x][y];
                p[0][x][y] = g;
                p[1][x][y] = g;
                p[2][x][y] = g;
            }
        }
        pixel = p;
        cs = Colorspace.RGB;
    }
    
    private void rgbToXyz() {
        assert (cs==Colorspace.RGB);
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float r = (pixel[0][x][y]+1)/2;
                float g = (pixel[1][x][y]+1)/2;
                float b = (pixel[2][x][y]+1)/2;
                pixel[0][x][y] =  2*(2.768892f*r + 1.751748f*g + 1.130200f*b)/5.65084f - 1;
                pixel[1][x][y] =  2*(1.000000f*r + 4.590700f*g + 0.060100f*b)/5.65084f - 1;
                pixel[2][x][y] =  2*(0.000000f*r + 0.056508f*g + 5.594292f*b)/5.65084f - 1;
            }
        }
        cs = Colorspace.XYZ;
    }
    
    private void xyzToRgb() {
        assert (cs==Colorspace.XYZ);
        
        for (int px=0; px<width; px++) {
            for (int py=0; py<height; py++) {
                float x = (pixel[0][px][py]+1)/2*5.65084f;
                float y = (pixel[1][px][py]+1)/2*5.65084f;
                float z = (pixel[2][px][py]+1)/2*5.65084f;
                pixel[0][px][py] = 2*( 0.418455000f*x - 0.158657000f*y - 0.0828349f*z) - 1;
                pixel[1][px][py] = 2*(-0.091164900f*x + 0.252426000f*y + 0.0157060f*z) - 1;
                pixel[2][px][py] = 2*( 0.000920857f*x - 0.002549750f*y + 0.1785950f*z) - 1;
            }
        }
        cs = Colorspace.RGB;
    }
    
    private void rgbToHsv() {
        assert (cs==Colorspace.RGB);
        
        for (int px=0; px<width; px++) {
            for (int py=0; py<height; py++) {
                float r = (pixel[0][px][py]+1)/2;
                float g = (pixel[1][px][py]+1)/2;
                float b = (pixel[2][px][py]+1)/2;
                float min = r;
                float max = r;
                if (min>g) {
                    min = g;
                }
                if (min>b) {
                    min = b;
                }
                if (max<g) {
                    max = g;
                }
                if (max<b) {
                    max = b;
                }
                
                float h = 0;
                if (max==min) {
                    h = 0;
                } else if (max==r) {
                    h = 60 * (0 + (g-b)/(max-min));
                } else if (max==g) {
                    h = 60 * (2 + (b-r)/(max-min));
                } else if (max==b) {
                    h = 60 * (4 + (r-g)/(max-min));
                }
                if (h<0) {
                    h+=360;
                }
                float s = (max==0) ? 0 : ((max-min)/max);
                float v = max;
                
                pixel[0][px][py] = 2*(h/360)-1;
                pixel[1][px][py] = 2*s-1;
                pixel[2][px][py] = 2*v-1;
            }
        }
        cs = Colorspace.HSV;
    }
    
    private void hsvToRgb() {
        assert (cs==Colorspace.HSV);
        
        for (int px=0; px<width; px++) {
            for (int py=0; py<height; py++) {
                float h = (pixel[0][px][py]+1)/2 * 360;
                float s = (pixel[1][px][py]+1)/2;
                float v = (pixel[2][px][py]+1)/2;
                int hi = (int)(h/60);
                float f = h/60 - hi;
                float p = 2*v*(1-s)-1;
                float q = 2*v*(1-s*f)-1;
                float t = 2*v*(1-s*(1-f))-1;
                v = 2*v - 1;
                switch (hi) {
                    case 0: case 6:
                        pixel[0][px][py]=v; pixel[1][px][py]=t; pixel[2][px][py]=p;
                        break;
                    case 1:
                        pixel[0][px][py]=q; pixel[1][px][py]=v; pixel[2][px][py]=p;
                        break;
                    case 2:
                        pixel[0][px][py]=p; pixel[1][px][py]=v; pixel[2][px][py]=t;
                        break;
                    case 3:
                        pixel[0][px][py]=p; pixel[1][px][py]=q; pixel[2][px][py]=v;
                        break;
                    case 4:
                        pixel[0][px][py]=t; pixel[1][px][py]=p; pixel[2][px][py]=v;
                        break;
                    case 5:
                        pixel[0][px][py]=v; pixel[1][px][py]=p; pixel[2][px][py]=q;
                        break;
                        
                }
            }
        }
        cs = Colorspace.RGB;
    }
    
    private void hsvToCssv() {
        assert (cs==Colorspace.HSV);
        
        float[][][] p = new float[4][width][height];
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                p[0][x][y] = (float)Math.cos(Math.PI*(1+pixel[0][x][y]));
                p[1][x][y] = (float)Math.sin(Math.PI*(1+pixel[0][x][y]));
                p[2][x][y] = pixel[1][x][y];
                p[3][x][y] = pixel[2][x][y];
            }
        }
        
        pixel = p;
        cs = Colorspace.CSSV;
    }
    
    private void cssvToHsv() {
        float[][][] p = new float[3][width][height];
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float c = pixel[0][x][y];
                float s = pixel[1][x][y];
                float a = (float)Math.atan2(s, c);
                while (a<0) {
                    a+=2*Math.PI;
                }
                p[0][x][y] = (a/(float)Math.PI) - 1;
                p[1][x][y] = pixel[2][x][y];
                p[2][x][y] = pixel[3][x][y];
            }
        }
        
        pixel = p;
        cs = Colorspace.HSV;
    }
    
    private void hsvToCss() {
        assert (cs==Colorspace.HSV);
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                
                float cos = (float)Math.cos(Math.PI*(1+pixel[0][x][y]));
                float sin = (float)Math.sin(Math.PI*(1+pixel[0][x][y]));
                float sat = pixel[1][x][y];
                float val = (pixel[2][x][y]+1)/2;
                pixel[0][x][y] = (2*cos*val) - 1;
                pixel[1][x][y] = (2*sin*val) - 1;
                pixel[2][x][y] = sat;
            }
        }
        cs = Colorspace.CSS;
    }
    
    private void cssToHsv() {
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float cosv = (pixel[0][x][y]+1)/2;
                float sinv = (pixel[1][x][y]+1)/2;
                float sat = pixel[2][x][y];
                
                float v = (float)Math.sqrt(cosv*cosv+sinv*sinv);
                
                float a = (float)Math.atan2(sinv, cosv);
                while (a<0) {
                    a+=2*Math.PI;
                }
                float h = (a/(float)Math.PI) - 1;
                
                
                pixel[0][x][y] = h;
                pixel[1][x][y] = sat;
                pixel[2][x][y] = 2*v-1;
            }
        }
        cs = Colorspace.HSV;
    }
    
    private void rgbToCmyk() {
        assert (cs==Colorspace.RGB);
        
        float[][][] p = new float[4][width][height];
        
        for (int x=0; x<width; x++) {
            for (int y=0; y<height; y++) {
                float cyan    = 1 - (pixel[0][x][y]+1)/2;
                float magenta = 1 - (pixel[1][x][y]+1)/2;
                float yellow  = 1 - (pixel[2][x][y]+1)/2;
                float k = min(cyan, min(magenta, yellow));
                p[0][x][y] = (k==1) ? 0 : (2*(cyan-k)/(1-k)-1);
                p[1][x][y] = (k==1) ? 0 : (2*(magenta-k)/(1-k)-1);
                p[2][x][y] = (k==1) ? 0 : (2*(yellow-k)/(1-k)-1);
                p[3][x][y] = 2*k-1;
            }
        }
        pixel = p;
        cs = Colorspace.CMYK;
    }
    
    private void cmykToRgb() {
        assert (cs==Colorspace.CMYK);
        
        float[][][] p = new float[3][width][height];
        
        for (int px=0; px<width; px++) {
            for (int py=0; py<height; py++) {
                float c = (pixel[0][px][py]+1)/2;
                float m = (pixel[1][px][py]+1)/2;
                float y = (pixel[2][px][py]+1)/2;
                float k = (pixel[3][px][py]+1)/2;
                float cyan    = c * (1-k) + k;
                float magenta = m * (1-k) + k;
                float yellow  = y * (1-k) + k;
                p[0][px][py]  = (1-cyan)*2 - 1;
                p[1][px][py]  = (1-magenta)*2 - 1;
                p[2][px][py]  = (1-yellow)*2 - 1;
            }
        }
        pixel = p;
        cs = Colorspace.RGB;
    }
    
    /**
     * @param channel color
     * @param x coordinate
     * @param y coordinate
     * @return the value of a pixel's channel
     */
    public float get(int channel, int x, int y) {
        return pixel[channel][x][y];
    }
    
    /**
     * Modifies a pixel's value.
     * @param channel the channel
     * @param x coordinate
     * @param y coordinate
     * @param val new value
     */
    public void set(int channel, int x, int y, float val) {
        pixel[channel][x][y] = val;
    }
    
    /**
     * @param rgb color code
     * @return the R component as a float in [-1;1]
     */
    public static float getR(int rgb) {
        return ((rgb>>16) & 0xFF) / 127.5f - 1;
    }
    
    /**
     * @param rgb color code
     * @return the G component as a float in [-1;1]
     */
    public static float getG(int rgb) {
        return ((rgb>>8) & 0xFF) / 127.5f - 1;
    }
    
    /**
     * @param rgb color code
     * @return the B component as a float in [-1;1]
     */
    public static float getB(int rgb) {
        return (rgb & 0xFF) / 127.5f - 1;
    }
    
    /**
     * Converts converts a float (between -1 and +1) to an int between 0 and 255.
     * @param f the float
     * @return an int
     */
    public static int toInt(float f) {
        return (int)(255 * (f+1) / 2);
    }
    
    /**
     * Converts an int between 0 and 255 to a float between -1 and +1.
     * @param i the integer
     * @return a float
     */
    public static float toFloat(int i) {
        return 2 * (i/255.0f) - 1;
    }
    
    /**
     * @param r red component
     * @param g green component
     * @param b blue component
     * @return the RGB code corresponding to three floats encoding a RGB color
     */
    public static int getRGB(float r, float g, float b) {
        return (toInt(r)<<16) | (toInt(g)<<8) | toInt(b);
    }
    
    /**
     * @return the image width
     */
    public int getWidth() {
        return width;
    }
    
    /**
     * @return the image height
     */
    public int getHeight() {
        return height;
    }
    
    /**
     * @return the number of channels
     */
    public int getDepth() {
        return cs.depth;
    }
    
    /**
     * @return the colorspace
     */
    public Colorspace getColorspace() {
        return cs;
    }
    
    /**
     * Normalizes the values in [-1,1].
     */
    public void normalize() {
        float min = Float.POSITIVE_INFINITY;
        float max = Float.NEGATIVE_INFINITY;
        for (int c=0; c<cs.depth; c++) {
            for (int x=0; x<width; x++) {
                for (int y=0; y<height; y++) {
                    if (get(c, x, y)<min) {
                        min = get(c,x,y);
                    }
                    if (get(c, x, y)>max) {
                        max = get(c,x,y);
                    }
                }
            }
        }
        for (int c=0; c<cs.depth; c++) {
            for (int x=0; x<width; x++) {
                for (int y=0; y<height; y++) {
                    float v = get(c, x, y);
                    set(c, x, y, 2*(v-min)/(max-min)-1);
                }
            }
        }
    }
    
    /**
     * Scales the image by an integer factor.
     * @param factor integer zoom factor
     * @return a new image
     */
    public Image getScaled(int factor) {
        Image res = new Image(factor*getWidth(), factor*getHeight(), cs);

        for (int x=0; x<res.getWidth(); x++) {
            int px = x/factor;
            for (int y=0; y<res.getHeight(); y++) {
                int py = y/factor;
                for (int z=0; z<getDepth(); z++) {
                    res.set(z, x, y, get(z, px, py));
                }
            }
        }

        return res;
    }
}
