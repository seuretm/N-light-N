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

import de.erichseifert.gral.data.DataSeries;
import de.erichseifert.gral.data.DataTable;
import de.erichseifert.gral.data.filters.Convolution;
import de.erichseifert.gral.data.filters.Filter;
import de.erichseifert.gral.data.filters.Kernel;
import de.erichseifert.gral.data.filters.Median;
import de.erichseifert.gral.graphics.Insets2D;
import de.erichseifert.gral.graphics.Label;
import de.erichseifert.gral.graphics.Location;
import de.erichseifert.gral.plots.XYPlot;
import de.erichseifert.gral.plots.lines.DefaultLineRenderer2D;
import de.erichseifert.gral.plots.lines.LineRenderer;
import de.erichseifert.gral.plots.points.DefaultPointRenderer2D;
import de.erichseifert.gral.plots.points.PointRenderer;
import de.erichseifert.gral.ui.InteractivePanel;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.image.BufferedImage;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;

/**
 * This is for plotting graphs nicely and easily.
 * @author Michele Alberti
 */
public class Tracer {
    /**
     * Expected number of samples for the plot. Used to filter incoming data to save memory
     */
    private final int EXPECTEDSAMPLES;
    /**
     * Current counter for filtering the samples
     */
    private int filterCount;
    /**
     * Total amount of points to be displayed
     */
    private int MAXPOINTS = 1000;
    /**
     * The plot object
     */
    private XYPlot plot;
    /**
     * Raw source data for the plot
     */
    private ArrayList<Point> data;
    /**
     * Raw source data reduce to maxPoints amount
     */
    private DataTable dataReduced;
    /**
     * List of colors for the traces in the graph
     */
    private static final Color[] color = {
            new Color(0.0f, 0.3f, 1.0f, 0.3f), //0
            new Color(1.0f, 0.0f, 0.0f, 0.3f), //1
            new Color(1.0f, 0.0f, 0.0f, 1.0f), //2
            new Color(0.0f, 0.0f, 1.0f, 1.0f), //3
            new Color(0.0f, 1.0f, 1.0f, 1.0f), //4
            new Color(0.0f, 0.0f, 0.0f, 0.5f), //5
    };

    /**
     * Specifies whether the plot will be shown or not
     */
    private boolean visible;
    /**
     * The frame used for the plot
     */
    private JFrame frame;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Creates an instance of Tracer. This object can be later populated through addPoint(x,y)
     * and finally a graph can be produced out of this data.
     * @param title title that the JFrame will have
     * @param xLabel label on the X axis
     * @param yLabel label on the Y axis
     * @param expectedSamples the number of expected samples that will be added to the plot. This is used to
     *                        perform a correct online down sampling of data maintaining the density of points.
     * @param visibile true if the the tracer has to be displayed
     */
    public Tracer(String title, String xLabel, String yLabel, int expectedSamples, boolean visibile) {

        // Save class variables
        this.EXPECTEDSAMPLES = expectedSamples;
        this.visible = visibile;

        // Init the down sampling filter so that he adds the fist point
        this.filterCount = EXPECTEDSAMPLES / MAXPOINTS;

        // Init the raw source data
        data = new ArrayList<>();

        if (visibile) {
            // Create the plot
            plot = new XYPlot();

            // Style the plot
            double insetsTop = 20.0,
                    insetsLeft = 80.0,
                    insetsBottom = 60.0,
                    insetsRight = 200.0;

            plot.setInsets(new Insets2D.Double(insetsTop, insetsLeft, insetsBottom, insetsRight));
            plot.getTitle().setText(title);

            // Style the plot area
            plot.getPlotArea().setBorderColor(color[0]);
            plot.getPlotArea().setBorderStroke(new BasicStroke(3f));

            // Style axes
            plot.getAxisRenderer(XYPlot.AXIS_X).setLabel(new Label(xLabel));
            plot.getAxisRenderer(XYPlot.AXIS_Y).setLabel(new Label(yLabel));
            plot.getAxisRenderer(XYPlot.AXIS_X).setIntersection(-Double.MAX_VALUE);
            plot.getAxisRenderer(XYPlot.AXIS_Y).setIntersection(-Double.MAX_VALUE);
            plot.getAxis("x").setMin(0);
            plot.getAxis("y").setMin(0);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * This method add a point to the raw source data for the plot
     * @param x coordinate of the point
     * @param y coordinate of the point
     */
    public void addPoint(double x, double y) {
        // Track the number of point tried to be added
        filterCount++;

        // Apply the down sampling
        if (filterCount >= EXPECTEDSAMPLES / MAXPOINTS) {
            // Add the point
            data.add(new Point(x, y));
            // Reset counter
            filterCount = 0;
        }

    }

    /**
     * Add to the plot the down sampled raw data.
     *
     * RED SEMI-TRANSPARENT DOTS
     */
    public void addRawData() {
        if (!visible) {
            return;
        }
        
        // Check that reduced data is available
        if (dataReduced == null) {
            reduceData();
        }

        // Add data to the plot
        DataSeries dsDataReduced = new DataSeries("Raw data entries", dataReduced);
        plot.add(dsDataReduced);

        // Style reduced data series
        PointRenderer points = new DefaultPointRenderer2D();
        points.setShape(new Ellipse2D.Double(-3.0, -3.0, 6.0, 6.0));
        points.setColor(color[1]);
        plot.setPointRenderers(dsDataReduced, points);
    }


    /**
     * Add the cumulated average of the raw data. It is computed using all points provided, even though it is displayed
     * only trough MAXPOINTS points.
     *
     * RED LINE
     */

    public void addCumulatedAverage() {
        if (!visible) {
            return;
        }
        // Check that reduced data is available
        if (dataReduced == null) {
            reduceData();
        }

        // Variable that keeps track of the avg
        double avg = 0;

        DataTable dataCumAvg = new DataTable(Double.class, Double.class);

        // Initial point (otherwise avg/0 makes infinity!)
        avg += (Double) dataReduced.getRow(0).get(1);
        dataCumAvg.add(dataReduced.getRow(0));

        // Compute cumulated average
        for (int i = 1; i < dataReduced.getRowCount(); i++) {
            avg += (Double) dataReduced.getRow(i).get(1);
            dataCumAvg.add(dataReduced.getRow(i).get(0), avg / (i + 1));
        }

        // Add data to the plot
        DataSeries dsCumAvg = new DataSeries("Cumulated average", dataCumAvg);
        plot.add(dsCumAvg);

        // Style average data series
        plot.setPointRenderers(dsCumAvg, null);
        LineRenderer lines = new DefaultLineRenderer2D();
        lines.setColor(color[2]);
        plot.setLineRenderers(dsCumAvg, lines);
    }

    /**
     *  Add the moving average of the raw data.
     *
     *  BLUE LINE
     */

    public void addMovingAverage() {
        if (!visible) {
            return;
        }
        // Check that reduced data is available
        if (dataReduced == null) {
            reduceData();
        }

        // Compute the moving average. The kernel size specifies the size of the 'moving' part
        int kernelMovingAverageSize = Math.round(MAXPOINTS / 20);
        Kernel kernelMovingAverage = Kernel.getUniform(
                kernelMovingAverageSize,
                kernelMovingAverageSize - 1,
                1.0D
        ).normalize();
        Convolution average = new Convolution(dataReduced, kernelMovingAverage, Filter.Mode.OMIT, new int[]{1});
        DataSeries dataMovingAvg = new DataSeries("Moving Average", average, new int[]{0, 1});

        // Add data to the plot
        plot.add(dataMovingAvg);

        // Style moving average data series
        plot.setPointRenderers(dataMovingAvg, null);
        LineRenderer lines = new DefaultLineRenderer2D();
        lines.setColor(color[3]);
        plot.setLineRenderers(dataMovingAvg, lines);
    }

    /**
     *  Add the moving median of the raw data.
     *
     *  CYAN LINE
     */

    public void addMovingMedian() {
        if (!visible) {
            return;
        }
        // Check that reduced data is available
        if (dataReduced == null) {
            reduceData();
        }

        // Compute the moving median. The kernel size specifies the size of the 'moving' part
        int kernelMovingMedianSize = Math.round(MAXPOINTS / 20);
        Median median = new Median(
                dataReduced,
                kernelMovingMedianSize,
                kernelMovingMedianSize - 1,
                Filter.Mode.OMIT,
                new int[]{1}
        );
        DataSeries dataMovingMedian = new DataSeries("Moving Median", median, new int[]{0, 1});

        // Add data to the plot
        plot.add(dataMovingMedian);

        // Style median data series
        plot.setPointRenderers(dataMovingMedian, null);
        LineRenderer lines = new DefaultLineRenderer2D();
        lines.setColor(color[4]);
        plot.setLineRenderers(dataMovingMedian, lines);
    }

    /**
     * Create the plot with the existing added data sets.
     */
    public void display() {
        if (!visible) {
            return;
        }

        frame = new JFrame();

        // Display the plot
        frame.getContentPane().add(new InteractivePanel(plot));

        // Display on screen
        frame.getContentPane().add(new InteractivePanel(plot), BorderLayout.CENTER);
        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.setMinimumSize(frame.getContentPane().getMinimumSize());
        frame.setSize(1024, 768);
        frame.setVisible(true);

        // Legend
        plot.setLegendVisible(true);
        plot.setLegendLocation(Location.EAST);
    }

    /**
     * Saves the plot on disk as png or text, depending on the extension of the file name
     * @param fName path of where the plot will be saved comprehensive of file name
     */
    public void savePlot(final String fName) {

        // Check whether the path is existing, if not create it
        File file = new File(fName);
        if (!file.isDirectory()) {
            file = file.getParentFile();
        }
        if (!file.exists()) {
            file.mkdirs();
        }

        // Dump raw data on file
        try {
            // Appends
            PrintWriter pr = new PrintWriter(new BufferedWriter(new FileWriter(fName + ".tracer", true)));

            for (Point p : data) {
                pr.println(p.x + "," + p.y);
            }
            pr.close();

        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("No such file exists.");
        }

        if (visible) {
            // Give time to the thread responsible to paint the plot
            // Measured-by-trial on my PC, might be increased on slower PCs
            try {
                Thread.sleep(2000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            try {
                ImageIO.write(getScreenShot(frame), "png", new File(fName + ".png"));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Private
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Perform an offline down sampling. Take the raw source data and reduces it to a set of size MAXPOINTS.
     */
    private void reduceData() {

        // Get the number of raw points
        int numRawSamples = data.size();

        // Check if we don't have enough points, just take all of them
        if (numRawSamples < MAXPOINTS) {
            MAXPOINTS = numRawSamples;
        }

        // Create the down sampled dataset
        dataReduced = new DataTable(Double.class, Double.class);

        /* Create the new reduced set. I use a double as iterating variable to handle cases where numRawSamples
         * is not a direct multiplier of MAXPOINTS.
         */
        for (double i = 0; i < numRawSamples; i += numRawSamples / MAXPOINTS) {
            dataReduced.add(data.get((int) (i + 0.5)).x, data.get((int) (i + 0.5)).y);
        }

    }

    /**
     * Get the image of a component (NOT a JComponent !!)
     * @param component the target component of which we want the image of
     * @return the image of the target component
     */
    private BufferedImage getScreenShot(Component component) {

        BufferedImage image = new BufferedImage(
                component.getWidth(),
                component.getHeight(),
                BufferedImage.TYPE_INT_RGB
        );

        // call the Component's paint method, using the Graphics object of the image.
        component.paint(image.getGraphics());
        return image;
    }


    /**
     * Support class for easier data structure modelling.
     * This is necessary as the data structure of GRAL is super heavy and I can't store millions of samples.
     * With this simple class and an arrayList however, I can.
     */
    private class Point {
        /**
         * X coordinate of the point
         */
        public final double x;
        /**
         * Y coordinate of the point
         */
        public final double y;

        public Point(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }
}

