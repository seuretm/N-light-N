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

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.SeriesRenderingOrder;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataItem;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.ui.RefineryUtilities;
import org.jfree.util.ShapeUtilities;

import javax.swing.*;
import java.awt.*;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.ListIterator;

/**
 * This is for plotting graphs nicely and easily.
 * @author Michele Alberti
 */
public class Tracer {

    /**
     * List of colors for the traces in the graph
     */
    private static final Color[] color = {
            new Color(1.0f, 0.4f, 0.4f, 0.6f), //1
            new Color(0.3f, 0.6f, 1.0f, 0.6f), //2
            new Color(0.1f, 0.6f, 0.3f, 0.6f), //3
            new Color(0.0f, 0.0f, 0.0f, 1.0f), //4
    };
    /**
     *  File name where the plots and the tracer txt file will be saved
     */
    private final String fName;
    /**
     *  Buffered writer used to write all the points on file (even if n < N)
     */
    private final BufferedWriter BW;
    /**
     * Dataset series: raw data, avg and moving median
     */
    private final XYSeries rawSeries = new XYSeries("Raw Data");
    private final XYSeries avgSeries = new XYSeries("Cumulative avg");
    private final XYSeries medianSeries = new XYSeries("Median");
    /**
     * Counter (n) and max counter (N) to filter points. We add only one points every N ones.
     * If N = 1 all points passed are added
     */
    private int n = 1;
    private int N;
    /**
     * Specifies whether the plot is desired or not. In case of "false" only the txt file will be generated
     */
    private boolean makePlot = false;
    /**
     * Size of the kernel for computing the moving median
     */
    private int KERNEL_SIZE = 10;
    /**
     * Chart of the plot.
     */
    private JFreeChart chart;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Constructor
    ///////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * Build a Tracer passing only the output file name. In fact, if no setupPlot() is called there will be
     * no graphical component initiated.
     *
     * @param fName output file name
     */
    public Tracer(String fName) throws IOException {
        this.fName = fName;
        this.BW = new BufferedWriter(new FileWriter(new File(fName + ".txt"), true));
    }

    /**
     * Creates an instance of Tracer. This object can be later populated through addPoint(x,y)
     * and finally a graph can be produced out of this data.
     * @param title title that the JFrame will have
     * @param xLabel label on the X axis
     * @param yLabel label on the Y axis
     */
    public void setupPlot(String title, String xLabel, String yLabel, int N, boolean visible) {

        // Flag that a plot is desired
        makePlot = true;

        // Init the raw source data
        this.N = N;

        // Create the dataset
        XYSeriesCollection xySeriesCollection = new XYSeriesCollection();

        // Add the series
        xySeriesCollection.addSeries(rawSeries);
        //xySeriesCollection.addSeries(avgSeries);
        xySeriesCollection.addSeries(medianSeries);

        chart = ChartFactory.createXYLineChart(title, xLabel, yLabel, xySeriesCollection, PlotOrientation.VERTICAL, true, true, false);

        // Handle the scatter & line plotter on the same chart
        XYPlot plot = (XYPlot) chart.getPlot();
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer();

        // "0" is the scatter plot (raw data)
        renderer.setSeriesLinesVisible(0, false);
        renderer.setSeriesShapesVisible(0, true);
        renderer.setSeriesShape(0, ShapeUtilities.createDiagonalCross(1, 0.3f));
        renderer.setSeriesPaint(0, color[1]);

        // "1" is the line plot (median)
        renderer.setSeriesLinesVisible(1, true);
        renderer.setSeriesShapesVisible(1, false);
        renderer.setSeriesPaint(1, color[3]);

        plot.setRenderer(renderer);
        plot.setSeriesRenderingOrder(SeriesRenderingOrder.REVERSE);

        // Visualize the plot
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(2000, 1000));

        if (visible) {
            JFrame frame = new JFrame();
            frame.setContentPane(chartPanel);
            frame.pack();
            RefineryUtilities.centerFrameOnScreen(frame);
            frame.setVisible(true);
        }

    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Public
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * This method add a point to the plot
     * @param x coordinate of the point
     * @param y coordinate of the point
     */
    public synchronized void addPoint(double x, double y) {

        // Save point on file
        try {
            BW.write(x + "\t" + y + "\n");
            BW.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // If the plot is desired
        if (makePlot) {
            // Enough samples were provided
            if (n == N) {

                // Add the point to the raw data series
                rawSeries.add(x, y);

                // Prepare array for computing the median
                double[] medianArray = new double[KERNEL_SIZE];

                // Get the iterator on the last item of the list, used for reverse exploration of the list of points
                ListIterator<XYDataItem> li = rawSeries.getItems().listIterator(rawSeries.getItemCount());

                // Iterate backwards to maximum KERNEL_SIZE elements
                double avgY = 0;
                double avgX = 0;
                int i = 0;
                while (li.hasPrevious() && i < KERNEL_SIZE) {

                    XYDataItem p = li.previous();

                    // Integrate average
                    avgX += p.getXValue();
                    avgY += p.getYValue();

                    // Fill the array for the median
                    medianArray[i] = p.getYValue();

                    // Increase counter
                    i++;
                }

                // Add the point to the cumulative average series
                avgSeries.add(avgX / i, avgY / i);

                // Resize the median array if necessary
                if (i < KERNEL_SIZE - 1) {
                    medianArray = Arrays.copyOf(medianArray, i);
                }

                // Add the point to the median series
                medianSeries.add(avgX / i, computeMedian(medianArray));

                // Reset the counter
                n = 1;

                // Adapt kernel size
                KERNEL_SIZE = 10 + rawSeries.getItemCount() / 20;

            } else {
                // Increment counter
                n++;
            }
        }
    }

    /**
     * Save the plot to file
     */
    public synchronized void savePlot() throws IOException {
        // Only available if the plot is prepared ofc
        if (makePlot) {
            File XYChart = new File(fName + ".jpeg");
            ChartUtilities.saveChartAsJPEG(XYChart, chart, 2000, 1000);
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // Private
    ///////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Compute the median of the given array
     * @param a the array to compute the median on
     * @return the median of the array
     */
    private double computeMedian(final double[] a) {
        Arrays.sort(a);
        if (a.length % 2 == 0) {
            return (a[a.length / 2] + a[a.length / 2 - 1]) / 2.0;
        } else {
            return a[a.length / 2];
        }
    }


}

