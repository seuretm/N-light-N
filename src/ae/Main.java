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


/**
 * This is the "main" of N-light-N when it is used as stand-alone application.
 * It will execute the different scripts passed as parameter to the command
 * line.
 * @author Mathias Seuret
 */
public class Main {
    /**
     * @param args the command line arguments
     * @throws Exception if some images cannot be loaded
     */
    public static void main(String[] args) throws Exception {

        // Check syntax
        if (args.length==0) {
            throw new IllegalArgumentException(
                    "Syntax: java -jar thejarfile.jar xml-script.xml [id=value]*"
            );
        }

        // Create script
        XMLScript script = new XMLScript(args[0]);

        // Parse aliases (if any)
        for (int i = 1; i < args.length; i++) {
            String[] parts = args[i].split("=");
            if (parts.length != 2) System.out.println("Syntax: id=value");
            script.definitions.put(parts[0], parts[1]);
        }

        // Start
        long start = System.currentTimeMillis();

        String res = script.execute();

        System.out.println("Script exited correctly with value " + res);

        XMLScript.printDuration(start);
    }
}
