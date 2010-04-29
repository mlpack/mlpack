/* MLPACK 0.2
 *
 * Copyright (c) 2008, 2009 Alexander Gray,
 *                          Garry Boyer,
 *                          Ryan Riegel,
 *                          Nikolaos Vasiloglou,
 *                          Dongryeol Lee,
 *                          Chip Mappus, 
 *                          Nishant Mehta,
 *                          Hua Ouyang,
 *                          Parikshit Ram,
 *                          Long Tran,
 *                          Wee Chin Wong
 *
 * Copyright (c) 2008, 2009 Georgia Institute of Technology
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation; either version 2 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA
 * 02110-1301, USA.
 */
/**
 * @file dataset_feature.h
 *
 * The DatasetFeature class, used by the Dataset class.
 * 
 * @bug These routines fail when trying to read files linewise that use the Mac
 * eol '\r'.  Both Windows and Unix eol ("\r\n" and '\n') work.  Use the
 * programs 'dos2unix' or 'tr' to convert the '\r's to '\n's.
 *
 */

#ifndef DATA_DATASET_FEATURE_H
#define DATA_DATASET_FEATURE_H

#include "../col/col_string.h"
#include "../la/matrix.h"
#include "../math/discrete.h"
#include "../file/textfile.h"

/**
 * Metadata about a particular dataset feature (attribute).
 *
 * Supports nominal, continuous, and integer values.
 */
class DatasetFeature {
  public:

    /**
     * Supported feature types.
     */
    enum Type {
      CONTINUOUS, /** Real-valued data. */
      INTEGER,  /** Integer valued data. */
      NOMINAL  /** Discrete data, each of which has a "name". */
    };
  
  private:
    String name_; /** Name of the feature. */
    Type type_; /** Type of data this feature represents. */
    ArrayList<String> value_names_; /** If nominal, the names of each numbered value. */

    // TODO: remove this
    OBJECT_TRAVERSAL(DatasetFeature) {
      OT_OBJ(name_);
      //OT_OBJ(reinterpret_cast<int &>(type_));
      OT_ENUM_EXPERT(type_, int,
        OT_ENUM_VAL(CONTINUOUS)
        OT_ENUM_VAL(INTEGER)
        OT_ENUM_VAL(NOMINAL));
      OT_OBJ(value_names_);
    }

   /**
    * Initialization common to all features.
    *
    * @param name_in the name of the feature
    */ 
   void InitGeneral(const char *name_in) {
      name_.Copy(name_in);
      value_names_.Init();
   }

  public:
    /**
     * Initialize to be a continuous feature.
     *
     * @param name_in the name of the feature
     */
    void InitContinuous(const char *name_in) {
      InitGeneral(name_in);
      type_ = CONTINUOUS;
    }

    /**
     * Initializes to an integer type.
     *
     * @param name_in the name of the feature
     */
    void InitInteger(const char *name_in) {
      InitGeneral(name_in);
      type_ = INTEGER;
    }

    /**
     * Initializes to a nominal type.
     *
     * The value_names list starts empty, so you need to add the name of
     * each feature to this.  (The dataset reading functions will do this
     * for you).
     *
     * @param name_in the name of the feature
     */
    void InitNominal(const char *name_in) {
      InitGeneral(name_in);
      type_ = NOMINAL;
    }
  
    /**
     * Creates a text version of the value based on the type.
     *
     * Continuous parameters are printed in floating point, and integers
     * are shown as integers.  For nominal, the value_name(int(value)) is
     * shown.  NaN (missing data) is always shown as '?'.
     *
     * @param value the value to format
     * @param result this will be initialized to the formatted text
     */
    void Format(double value, String *result) const;
  
    /**
     * Parses a string into the particular value.
     *
     * Integers and continuous are parsed using the normal functions.
     * For nominal, the entry 
     *
     * If an invalid parse occurs, such as a mal-formatted number or
     * a nominal value not in the list, SUCCESS_FAIL will be returned.
     *
     * @param str the string to parse
     * @param d where to store the result
     */
    success_t Parse(const char *str, double& d) const;
  
    /**
     * Gets what the feature is named.
     *
     * @return the name of the feature; for point, "Age" or "X Position"
     */
    const String& name() const {
      return name_;
    }
  
    /**
     * Identifies the type of feature.
     *
     * @return whether this is DatasetFeature::CONTINUOUS, INTEGER, or NOMINAL
     */
    Type type() const {
      return type_;
    }
  
    /**
     * Returns the name of a particular nominal value, given its index.
     *
     * The first nominal value is 0, the second is 1, etc.
     *
     * @param value the number of the value
     */
    const String& value_name(int value) const {
      DEBUG_ASSERT(type_ == NOMINAL);
      return value_names_[value];
    }
  
    /**
     * The number of nominal values.
     *
     * The values 0 to n_values() - 1 are valid.
     * This will return zero for CONTINUOUS and INTEGER types.
     *
     * @return the number of nominal values
     */
    index_t n_values() const {
      return value_names_.size();
    }
  
    /**
     * Gets the array of value names.
     *
     * Useful for creating a nominal feature yourself.
     *
     * @return a mutable array of value names
     */
    ArrayList<String>& value_names() {
      return value_names_;
    }
};

#endif
