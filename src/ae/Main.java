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
package ae;

import diuf.diva.dia.ms.script.XMLScript;

import java.io.IOException;

/**
 * @author Mathias Seuret
 */
public class Main {
    /**
     * @param args the command line arguments
     * @throws IOException if some images cannot be loaded
     */
    public static void main(String[] args) throws Exception {
        if (args.length==0) {
            throw new IllegalArgumentException(
                    "Syntax: java -jar thejarfile.jar xml-script.xml"
            );
        }
        for (String scriptName : args) {
            long start = System.currentTimeMillis();
            XMLScript script = new XMLScript(scriptName);
            String res = script.execute();
            System.out.println("Script exited correctly with value "+res);
            long chrono = System.currentTimeMillis() - start;
            long t = chrono / 1000;
            if (t<60) {
                System.out.println("Run time: "+t+" seconds");
            } else if (t<3600) {
                System.out.printf("Run time: %d:%02d\n", (t/60), (t%60));
            } else {
                System.out.printf("Run time: %d:%02d:%02d\n", (t/3600), ((t%3600)/60), (t%60));
            }
            script.clearData();
            System.gc();
            try {
                Thread.sleep(500);
            } catch (InterruptedException ignored) {
            }
        }
    }    
}
