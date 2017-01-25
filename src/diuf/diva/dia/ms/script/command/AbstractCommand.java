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

package diuf.diva.dia.ms.script.command;

import diuf.diva.dia.ms.script.XMLScript;
import org.jdom2.Element;

/**
 * This class provides some methods which simplify the act of parsing the XML script.
 * Furthermore using these methods ensure that each command gets correctly preprocessed and that
 * meaningful exceptions are thrown in case of problems.
 *
 * @author Mathias Seuret, Michele Alberti
 */
public abstract class AbstractCommand {
    protected XMLScript script;
    
    /**
     * Constructs a new command for the given script.
     * @param script script instance ("this" in the script)
     */
    public AbstractCommand(XMLScript script) {
        this.script = script;
    }

    /**
     * Read an attribute of a tag.
     *
     * @param e    the element from which we want to read
     * @param name the name of the attribute field
     * @return the value of the attribute
     */
    public String readAttribute(Element e, String name) {
        String result = e.getAttributeValue(name);
        if (result==null) {
            throw new Error(
                    tagName()+": Cannot find attribute "+name+" in element "+e.getName()
            );
        }
        return script.preprocess(result);
    }

    /**
     * This method return the text contained into the tag. Example <a>hello</a> returns "hello".
     * Note that this applies also if the element 'e' has children.
     *
     * @param e the element from which we want to read
     * @return the text value contained into the element
     */
    public String readElement(Element e) {
        String result = e.getText();
        if (result==null) {
            throw new Error(
                    tagName()+": Element "+e.getName()+" has no content."
            );
        }
        return script.preprocess(result);
    }

    /**
     * Return the content of the specified child of the parent element passed. It makes use of
     * readElement(Element e).
     * @param parent the parent element whose child will be read
     * @param child the specific child that must be read
     * @return the text content of the tag
     */
    public String readElement(Element parent, String child) {
        Element c = parent.getChild(child);
        if (c==null) {
            throw new Error(tagName()+": Cannot find the element <"+child+"> in <"+parent.getName()+">");
        }
        return readElement(c);
    }

    /**
     * This method is formatting and throwing a meaningful error
     * @param msg the content of the message
     */
    public void error(String msg) {
        throw new Error(tagName()+": "+msg);
    }

    /**
     * This methods is the body of every command. It gets called every time a specific command is
     * detected while parsing the XML script.
     * @param element root element of the command detected
     * @return result of the execution of the command. Might be a number or an error msg
     * @throws Exception some commands might make use of IO or other stuff that throws exceptions
     */
    public abstract String execute(Element element) throws Exception;

    /**
     * This is method should return the tag name so that it can be correctly detected while
     * parsing the XML.
     * @return the XML tag syntax. Example: "create-classifier" for CreateClassifier.java
     */
    public abstract String tagName();
}
